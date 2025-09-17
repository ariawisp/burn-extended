use burn_core as burn;

use burn::module::Module;
use burn_store::{ApplyResult, ModuleSnapshot, TensorSnapshot};
use burn_tensor::{DType, TensorData};
use half::bf16;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

type Result<T> = anyhow::Result<T>;

fn read_exact<R: Read>(mut r: R, buf: &mut [u8]) -> Result<()> {
    use anyhow::anyhow;
    r.read_exact(buf).map_err(|e| anyhow!(e))
}
fn read_u32<R: Read>(mut r: R) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(&mut r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_f32<R: Read>(mut r: R) -> Result<f32> {
    let mut b = [0u8; 4];
    read_exact(&mut r, &mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn align_up(off: u64, align: u64) -> u64 {
    if align == 0 { return off; }
    let rem = off % align;
    if rem == 0 { off } else { off + (align - rem) }
}

fn tensor_from_bf16_bytes(bytes: Vec<u8>, shape: Vec<usize>) -> TensorData {
    TensorData::from_bytes_vec(bytes, shape, DType::BF16)
}

fn snapshot(path: &str, data: TensorData) -> TensorSnapshot {
    TensorSnapshot::from_data(
        data,
        path.split('.').map(|s| s.to_string()).collect(),
        vec!["ModelBin".to_string()],
        burn_core::module::ParamId::new(),
    )
}

pub struct ModelBinHeader {
    pub num_blocks: u32,
    pub num_experts: u32,
    pub embedding_dim: u32,
    pub mlp_dim: u32,
    pub head_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub attention_window: u32,
}

pub struct TokenizerHeader {
    pub num_text: u32,
    pub num_special: u32,
    pub regex_size: u32,
    pub tokens_size: u32,
}

pub struct ModelBinOffsets {
    pub after_tokenizer: u64,
}

pub struct ParsedModelBin {
    pub header: ModelBinHeader,
    pub tokenizer: TokenizerHeader,
    pub after_tokenizer: u64,
}

fn parse_headers(file: &mut File) -> Result<ParsedModelBin> {
    use anyhow::bail;
    // Magic
    let mut magic = [0u8; 16];
    read_exact(&mut *file, &mut magic)?;
    if &magic[..12] != b"GPT-OSS v1.0" {
        bail!("invalid magic header");
    }
    // Model UUID
    let mut model_uuid = [0u8; 16];
    read_exact(&mut *file, &mut model_uuid)?;
    // Header fields
    let _context_length = read_u32(&mut *file)?;
    let num_blocks = read_u32(&mut *file)?;
    let num_experts = read_u32(&mut *file)?;
    let _num_active_experts = read_u32(&mut *file)?;
    let embedding_dim = read_u32(&mut *file)?;
    let mlp_dim = read_u32(&mut *file)?;
    let _swiglu_limit = read_f32(&mut *file)?;
    let head_dim = read_u32(&mut *file)?;
    let num_heads = read_u32(&mut *file)?;
    let num_kv_heads = read_u32(&mut *file)?;
    let attention_window = read_u32(&mut *file)?;
    let _rope_theta = read_f32(&mut *file)?;
    let _interpolation_scale = read_f32(&mut *file)?;
    let _yarn_offset = read_f32(&mut *file)?;
    let _yarn_scale = read_f32(&mut *file)?;
    let _yarn_multiplier = read_f32(&mut *file)?;
    let _rms_eps = read_f32(&mut *file)?;
    // Layout UUID
    let mut layout_uuid = [0u8; 16];
    read_exact(&mut *file, &mut layout_uuid)?;

    // Tokenizer header
    let mut tok_uuid = [0u8; 16];
    read_exact(&mut *file, &mut tok_uuid)?;
    let num_special = read_u32(&mut *file)?;
    let num_text = read_u32(&mut *file)?;
    let regex_size = read_u32(&mut *file)?;
    let tokens_size = read_u32(&mut *file)?;
    // Skip special UUID table
    let mut buf = vec![0u8; num_special as usize * 16];
    read_exact(&mut *file, &mut buf)?;
    // Read regex + NUL
    let mut regex = vec![0u8; regex_size as usize];
    read_exact(&mut *file, &mut regex)?;
    // Read token blob
    let mut tok_blob = vec![0u8; tokens_size as usize];
    read_exact(&mut *file, &mut tok_blob)?;
    // Align to 16KB
    let cur = file.seek(SeekFrom::Current(0))?;
    let aligned = align_up(cur, 16384);
    if aligned != cur {
        file.seek(SeekFrom::Start(aligned))?;
    }

    Ok(ParsedModelBin {
        header: ModelBinHeader {
            num_blocks,
            num_experts,
            embedding_dim,
            mlp_dim,
            head_dim,
            num_heads,
            num_kv_heads,
            attention_window,
        },
        tokenizer: TokenizerHeader {
            num_text,
            num_special,
            regex_size,
            tokens_size,
        },
        after_tokenizer: aligned,
    })
}

pub fn parse_modelbin(path: &Path) -> Result<ParsedModelBin> {
    let mut f = File::open(path)?;
    parse_headers(&mut f)
}

pub fn load_modelbin_into<B: burn::tensor::backend::Backend, M: Module<B> + Clone>(
    model: &mut M,
    path: &Path,
    validate: bool,
) -> Result<ApplyResult> {
    use anyhow::Context;
    let mut f = File::open(path).with_context(|| format!("open {:?}", path))?;
    let parsed = parse_headers(&mut f)?;

    let included_tokens = (parsed.tokenizer.num_text + parsed.tokenizer.num_special) as usize;
    let d_model = parsed.header.embedding_dim as usize;
    let n_layers = parsed.header.num_blocks as usize;
    let n_heads = parsed.header.num_heads as usize;
    let kv_heads = parsed.header.num_kv_heads as usize;
    let head_dim = parsed.header.head_dim as usize;

    let mut snapshots: Vec<TensorSnapshot> = Vec::new();

    // Embedding weight [tokens, d_model] BF16
    let emb_bytes = included_tokens * d_model * 2;
    let mut buf = vec![0u8; emb_bytes];
    read_exact(&mut f, &mut buf)?;
    snapshots.push(snapshot(
        "tok_emb.weight",
        tensor_from_bf16_bytes(buf, vec![included_tokens, d_model]),
    ));

    for l in 0..n_layers {
        // align 16 and read attn.norm.scale [d_model]
        let cur = f.seek(SeekFrom::Current(0))?;
        let need = align_up(cur, 16);
        if need != cur { f.seek(SeekFrom::Start(need))?; }
        let mut buf = vec![0u8; d_model * 2];
        read_exact(&mut f, &mut buf)?;
        snapshots.push(snapshot(&format!("layers.{l}.norm_attn.scale"), tensor_from_bf16_bytes(buf, vec![d_model])));

        // align 16 and read qkv (w+b)
        let cur = f.seek(SeekFrom::Current(0))?;
        let need = align_up(cur, 16);
        if need != cur { f.seek(SeekFrom::Start(need))?; }
        let rows = head_dim * (n_heads + 2 * kv_heads);
        let q_rows = n_heads * head_dim;
        let kv_rows = kv_heads * head_dim;
        // weight [rows, d_model]
        let mut wbuf = vec![0u8; rows * d_model * 2];
        read_exact(&mut f, &mut wbuf)?;
        // bias [rows]
        let mut bbuf = vec![0u8; rows * 2];
        read_exact(&mut f, &mut bbuf)?;
        // Split rows: Q, K, V
        let row_stride = d_model * 2;
        let mut qw = Vec::with_capacity(q_rows * d_model * 2);
        let mut kw = Vec::with_capacity(kv_rows * d_model * 2);
        let mut vw = Vec::with_capacity(kv_rows * d_model * 2);
        for r in 0..rows {
            let start = r * row_stride;
            let end = start + row_stride;
            if r < q_rows {
                qw.extend_from_slice(&wbuf[start..end]);
            } else if r < q_rows + kv_rows {
                kw.extend_from_slice(&wbuf[start..end]);
            } else {
                vw.extend_from_slice(&wbuf[start..end]);
            }
        }
        let mut qb = Vec::with_capacity(q_rows * 2);
        let mut kb = Vec::with_capacity(kv_rows * 2);
        let mut vb = Vec::with_capacity(kv_rows * 2);
        for r in 0..rows {
            let start = r * 2;
            let end = start + 2;
            if r < q_rows {
                qb.extend_from_slice(&bbuf[start..end]);
            } else if r < q_rows + kv_rows {
                kb.extend_from_slice(&bbuf[start..end]);
            } else {
                vb.extend_from_slice(&bbuf[start..end]);
            }
        }
        snapshots.push(snapshot(&format!("layers.{l}.attn.query.weight"), tensor_from_bf16_bytes(qw, vec![q_rows, d_model])));
        snapshots.push(snapshot(&format!("layers.{l}.attn.key.weight"), tensor_from_bf16_bytes(kw, vec![kv_rows, d_model])));
        snapshots.push(snapshot(&format!("layers.{l}.attn.value.weight"), tensor_from_bf16_bytes(vw, vec![kv_rows, d_model])));
        snapshots.push(snapshot(&format!("layers.{l}.attn.query.bias"), tensor_from_bf16_bytes(qb, vec![q_rows])));
        snapshots.push(snapshot(&format!("layers.{l}.attn.key.bias"), tensor_from_bf16_bytes(kb, vec![kv_rows])));
        snapshots.push(snapshot(&format!("layers.{l}.attn.value.bias"), tensor_from_bf16_bytes(vb, vec![kv_rows])));

        // sinks [n_heads] -> reshape done by module when loaded
        let cur = f.seek(SeekFrom::Current(0))?;
        let need = align_up(cur, 16);
        if need != cur { f.seek(SeekFrom::Start(need))?; }
        let mut sbuf = vec![0u8; n_heads * 2];
        read_exact(&mut f, &mut sbuf)?;
        snapshots.push(snapshot(&format!("layers.{l}.attn.sinks"), tensor_from_bf16_bytes(sbuf, vec![n_heads])));

        // attn.out (w+b)
        let cur = f.seek(SeekFrom::Current(0))?;
        let need = align_up(cur, 16);
        if need != cur { f.seek(SeekFrom::Start(need))?; }
        let mut ow = vec![0u8; d_model * d_model * 2];
        read_exact(&mut f, &mut ow)?;
        let mut ob = vec![0u8; d_model * 2];
        read_exact(&mut f, &mut ob)?;
        snapshots.push(snapshot(&format!("layers.{l}.attn.output.weight"), tensor_from_bf16_bytes(ow, vec![d_model, d_model])));
        snapshots.push(snapshot(&format!("layers.{l}.attn.output.bias"), tensor_from_bf16_bytes(ob, vec![d_model])));

        // mlp.norm.scale
        let cur = f.seek(SeekFrom::Current(0))?;
        let need = align_up(cur, 16);
        if need != cur { f.seek(SeekFrom::Start(need))?; }
        let mut mns = vec![0u8; d_model * 2];
        read_exact(&mut f, &mut mns)?;
        snapshots.push(snapshot(&format!("layers.{l}.mlp.norm.scale"), tensor_from_bf16_bytes(mns, vec![d_model])));

        // mlp.gate (w+b): weight [num_experts, d_model], bias [num_experts]
        let cur = f.seek(SeekFrom::Current(0))?;
        let need = align_up(cur, 16);
        if need != cur { f.seek(SeekFrom::Start(need))?; }
        let e = parsed.header.num_experts as usize;
        let mut gw = vec![0u8; e * d_model * 2];
        read_exact(&mut f, &mut gw)?;
        let mut gb = vec![0u8; e * 2];
        read_exact(&mut f, &mut gb)?;
        snapshots.push(snapshot(&format!("layers.{l}.mlp.gate.weight"), tensor_from_bf16_bytes(gw, vec![e, d_model])));
        snapshots.push(snapshot(&format!("layers.{l}.mlp.gate.bias"), tensor_from_bf16_bytes(gb, vec![e])));
    }

    // final norm.scale
    let cur = f.seek(SeekFrom::Current(0))?;
    let need = align_up(cur, 16);
    if need != cur { f.seek(SeekFrom::Start(need))?; }
    let mut fns = vec![0u8; d_model * 2];
    read_exact(&mut f, &mut fns)?;
    snapshots.push(snapshot("norm_final.scale", tensor_from_bf16_bytes(fns, vec![d_model])));

    // unembedding weight [tokens, d_model]
    let cur = f.seek(SeekFrom::Current(0))?;
    let need = align_up(cur, 16);
    if need != cur { f.seek(SeekFrom::Start(need))?; }
    let mut uw = vec![0u8; included_tokens * d_model * 2];
    read_exact(&mut f, &mut uw)?;
    snapshots.push(snapshot("lm_head.weight", tensor_from_bf16_bytes(uw, vec![included_tokens, d_model])));

    // Note: MoE expert weights follow; omitted in this initial reader.

    let result = model.apply(snapshots);
    if validate && !result.errors.is_empty() {
        anyhow::bail!("Import errors: {:?}", result.errors);
    }
    Ok(result)
}
