use burn_store::safetensors::SafetensorsError;
use burn_tensor::DType;

pub(crate) fn burn_dtype_from_safetensors(
    dtype: safetensors::Dtype,
) -> Result<DType, SafetensorsError> {
    use safetensors::Dtype as ST;
    Ok(match dtype {
        ST::F64 => DType::F64,
        ST::F32 => DType::F32,
        ST::F16 => DType::F16,
        ST::BF16 => DType::BF16,
        ST::I64 => DType::I64,
        ST::I32 => DType::I32,
        ST::I16 => DType::I16,
        ST::I8 => DType::I8,
        ST::U64 => DType::U64,
        ST::U32 => DType::U32,
        ST::U8 => DType::U8,
        ST::BOOL => DType::Bool,
        _ => {
            return Err(SafetensorsError::Other(format!(
                "Unsupported dtype: {:?}",
                dtype
            )))
        }
    })
}

pub(crate) fn elem_size(dtype: DType) -> usize {
    match dtype {
        DType::F64 | DType::I64 | DType::U64 => 8,
        DType::F32 | DType::I32 | DType::U32 | DType::Flex32 => 4,
        DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
        DType::I8 | DType::U8 | DType::Bool | DType::QFloat(_) => 1,
    }
}
