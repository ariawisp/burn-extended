use burn_core as burn;

use burn::module::Module;
use burn_store::{ApplyResult, ModuleSnapshot, TensorSnapshot};
use burn_tensor::{DType, TensorData};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use alloc::sync::Arc;

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
    pub num_active_experts: u32,
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
    let num_active_experts = read_u32(&mut *file)?;
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
            num_active_experts,
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

pub fn load_modelbin_into<B: burn::tensor::backend::Backend, M: Module<B> + Clone + 'static>(
    model: &mut M,
    path: &Path,
    validate: bool,
    skip_moe: bool,
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
    let ffn_hidden = parsed.header.mlp_dim as usize;

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

        // sinks stored as [n_heads], reshape to [kv_heads, groups]
        let cur = f.seek(SeekFrom::Current(0))?;
        let need = align_up(cur, 16);
        if need != cur { f.seek(SeekFrom::Start(need))?; }
        let mut sbuf = vec![0u8; n_heads * 2];
        read_exact(&mut f, &mut sbuf)?;
        let groups = n_heads / kv_heads;
        snapshots.push(snapshot(&format!("layers.{l}.attn.sinks"), tensor_from_bf16_bytes(sbuf, vec![kv_heads, groups])));

        // attn.out (w+b)
        let cur = f.seek(SeekFrom::Current(0))?;
        let need = align_up(cur, 16);
        if need != cur { f.seek(SeekFrom::Start(need))?; }
        let mut ow = vec![0u8; d_model * (n_heads * head_dim) * 2];
        read_exact(&mut f, &mut ow)?;
        let mut ob = vec![0u8; d_model * 2];
        read_exact(&mut f, &mut ob)?;
        snapshots.push(snapshot(&format!("layers.{l}.attn.output.weight"), tensor_from_bf16_bytes(ow, vec![d_model, n_heads * head_dim])));
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

    // If MoE is skipped, apply non-MoE snapshots and return
    if skip_moe {
        let result = model.apply(snapshots);
        if validate && !result.errors.is_empty() {
            anyhow::bail!("Import errors: {:?}", result.errors);
        }
        return Ok(result);
    }

    let e = parsed.header.num_experts as usize;
    let rows_mlp1 = 2 * ffn_hidden;
    let cols_mlp1 = d_model;
    let rows_mlp2 = d_model;
    let cols_mlp2 = ffn_hidden;
    let bpr_mlp1 = (cols_mlp1 + 1) / 2;
    let bpr_mlp2 = (cols_mlp2 + 1) / 2;
    let gpr_mlp1 = (cols_mlp1 + 31) / 32; // scales per 32-col block
    let bytes_mlp1_blocks = rows_mlp1 * bpr_mlp1;
    let bytes_mlp1_scales = rows_mlp1 * gpr_mlp1;
    let bytes_mlp1_bias = rows_mlp1 * 2;
    let gpr_mlp2 = (cols_mlp2 + 31) / 32;
    let bytes_mlp2_blocks = rows_mlp2 * bpr_mlp2;
    let bytes_mlp2_scales = rows_mlp2 * gpr_mlp2;
    let bytes_mlp2_bias = rows_mlp2 * 2;

    // Build per-layer streaming indices for MoE quantized weights (mmap-friendly offsets).
    let mut per_layer_expert_offsets: Vec<Vec<crate::moe::MoeExpertOffsets>> = Vec::with_capacity(n_layers);

    for l in 0..n_layers {
        // align 16KB before MoE layer payload
        let cur = f.seek(SeekFrom::Current(0))?;
        let need = align_up(cur, 16384);
        if need != cur { f.seek(SeekFrom::Start(need))?; }
        let mut layer_offsets: Vec<crate::moe::MoeExpertOffsets> = Vec::with_capacity(e);
        // Accumulate biases resident (BF16) while indexing quantized payloads.
        let mut mlp1_b_all = vec![0u8; e * bytes_mlp1_bias];
        let mut mlp2_b_all = vec![0u8; e * bytes_mlp2_bias];

        for ex in 0..e {
            // mlp1 blocks
            let cur = f.seek(SeekFrom::Current(0))?;
            let need = align_up(cur, 16);
            if need != cur { f.seek(SeekFrom::Start(need))?; }
            let mlp1_blocks_off = f.seek(SeekFrom::Current(0))?;
            // skip blocks
            f.seek(SeekFrom::Current(bytes_mlp1_blocks as i64))?;
            // mlp1 scales
            let cur = f.seek(SeekFrom::Current(0))?;
            let need = align_up(cur, 16);
            if need != cur { f.seek(SeekFrom::Start(need))?; }
            let mlp1_scales_off = f.seek(SeekFrom::Current(0))?;
            f.seek(SeekFrom::Current(bytes_mlp1_scales as i64))?;
            // mlp1 bias
            let cur = f.seek(SeekFrom::Current(0))?;
            let need = align_up(cur, 16);
            if need != cur { f.seek(SeekFrom::Start(need))?; }
            let mut bb = vec![0u8; bytes_mlp1_bias];
            read_exact(&mut f, &mut bb)?;
            let offb = ex * bytes_mlp1_bias;
            mlp1_b_all[offb..offb + bytes_mlp1_bias].copy_from_slice(&bb);

            // mlp2 blocks
            let cur = f.seek(SeekFrom::Current(0))?;
            let need = align_up(cur, 16);
            if need != cur { f.seek(SeekFrom::Start(need))?; }
            let mlp2_blocks_off = f.seek(SeekFrom::Current(0))?;
            f.seek(SeekFrom::Current(bytes_mlp2_blocks as i64))?;
            // mlp2 scales
            let cur = f.seek(SeekFrom::Current(0))?;
            let need = align_up(cur, 16);
            if need != cur { f.seek(SeekFrom::Start(need))?; }
            let mlp2_scales_off = f.seek(SeekFrom::Current(0))?;
            f.seek(SeekFrom::Current(bytes_mlp2_scales as i64))?;
            // mlp2 bias
            let cur = f.seek(SeekFrom::Current(0))?;
            let need = align_up(cur, 16);
            if need != cur { f.seek(SeekFrom::Start(need))?; }
            let mut bb2 = vec![0u8; bytes_mlp2_bias];
            read_exact(&mut f, &mut bb2)?;
            let off2b = ex * bytes_mlp2_bias;
            mlp2_b_all[off2b..off2b + bytes_mlp2_bias].copy_from_slice(&bb2);

            layer_offsets.push(crate::moe::MoeExpertOffsets {
                mlp1_blocks_off,
                mlp1_scales_off,
                mlp2_blocks_off,
                mlp2_scales_off,
            });
        }
        // Push biases only; quantized blocks/scales will be streamed from mmap
        snapshots.push(snapshot(
            &format!("layers.{l}.mlp.mlp1_bias"),
            tensor_from_bf16_bytes(mlp1_b_all, vec![e, rows_mlp1]),
        ));
        snapshots.push(snapshot(
            &format!("layers.{l}.mlp.mlp2_bias"),
            tensor_from_bf16_bytes(mlp2_b_all, vec![e, rows_mlp2]),
        ));
        per_layer_expert_offsets.push(layer_offsets);
    }

    // Apply all non-MoE and bias parameters first
    let result = model.apply(snapshots);
    if validate && !result.errors.is_empty() {
        anyhow::bail!("Import errors: {:?}", result.errors);
    }
    // Create mmap for the file and attach streaming contexts per layer if the model type supports it.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&f)? };
    // Advise random access to reduce readahead thrash on Apple/Linux
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    unsafe {
        let ptr = mmap.as_ptr() as *mut core::ffi::c_void;
        let len = mmap.len();
        let _ = libc::madvise(ptr, len, libc::MADV_RANDOM);
    }
    let mmap_arc = Arc::new(mmap);

    // Build contexts per layer
    let mut contexts: Vec<Arc<crate::moe::MoeStreamingContext>> = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let ctx = crate::moe::MoeStreamingContext {
            mmap: mmap_arc.clone(),
            experts: per_layer_expert_offsets[l].clone(),
            rows_mlp1,
            cols_mlp1,
            rows_mlp2,
            cols_mlp2,
            ue8_offset: 14,
        };
        contexts.push(Arc::new(ctx));
    }

    // If M is our GPT-OSS model, attach contexts; otherwise ignore.
    // We detect by downcasting via Any (Module doesn't expose RTTI), so expose a helper on the model instead.
    // Use trait bounds: try to call a known method via specialization is not possible; fall back to a runtime trick using cfg(feature) is overkill.
    // Instead, attempt to use public method via any: we define a trait locally and implement for model type.
    attach_moe_streaming_contexts_if_supported(model, contexts);

    Ok(result)
}

// Helper: attach contexts to GPT-OSS model if the concrete type exposes the method.
fn attach_moe_streaming_contexts_if_supported<B: burn::tensor::backend::Backend, M: Module<B> + Clone + 'static>(
    model: &mut M,
    contexts: Vec<Arc<crate::moe::MoeStreamingContext>>,
) {
    // Use Any downcast to GptOssModel if available without creating a direct dependency cycle.
    use core::any::Any;
    // Safety: Module<B> types are 'static here.
    if let Some(m_any) = (model as &mut dyn Any).downcast_mut::<crate::models::gpt_oss::GptOssModel<B>>() {
        m_any.set_moe_streaming_contexts(contexts);
    }
}

// Upload device-resident quantized MoE tensors per layer for faster runtime decoding.
