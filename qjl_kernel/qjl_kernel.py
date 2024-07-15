import torch
from qjl_kernel import cuda_qjl_quant
from qjl_kernel import cuda_qjl_score
from qjl_kernel import cuda_qjl_gqa_score


def qjl_quant(key_states, outlier_indices, rand_prj, outlier_sketch_dim):
    key_dtype = key_states.dtype
    rand_dtype = rand_prj.dtype

    if key_dtype == torch.half and rand_dtype == torch.half:
        return cuda_qjl_quant.qjl_quant_half_half(
            key_states, outlier_indices, rand_prj, outlier_sketch_dim)
    elif key_dtype == torch.half and rand_dtype == torch.float:
        return cuda_qjl_quant.qjl_quant_half_float(
            key_states, outlier_indices, rand_prj, outlier_sketch_dim)
    elif key_dtype == torch.float and rand_dtype == torch.float:
        return cuda_qjl_quant.qjl_quant_float_float(
            key_states, outlier_indices, rand_prj, outlier_sketch_dim)
    elif key_dtype == torch.bfloat16 and rand_dtype == torch.bfloat16:
        return cuda_qjl_quant.qjl_quant_bf16_bf16(
            key_states, outlier_indices, rand_prj, outlier_sketch_dim)
    elif key_dtype == torch.bfloat16 and rand_dtype == torch.float:
        return cuda_qjl_quant.qjl_quant_bf16_float(
            key_states, outlier_indices, rand_prj, outlier_sketch_dim)
    else:
        raise TypeError("Unsupported data types for QJL quantization")

def qjl_score(key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj):
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    if query_dtype == torch.half and rand_dtype == torch.half:
        return cuda_qjl_score.qjl_score_cuda_half_half(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    elif query_dtype == torch.half and rand_dtype == torch.float:
        return cuda_qjl_score.qjl_score_cuda_half_float(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    elif query_dtype == torch.float and rand_dtype == torch.float:
        return tcuda_qjl_score.qjl_score_cuda_float_float(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    elif query_dtype == torch.bfloat16 and rand_dtype == torch.bfloat16:
        return cuda_qjl_score.qjl_score_cuda_bf16_bf16(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    elif query_dtype == torch.bfloat16 and rand_dtype == torch.float:
        return cuda_qjl_score.qjl_score_cuda_bf16_float(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    else:
        raise TypeError("Unsupported data types for QJL score calculation")

def qjl_gqa_score(key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj):
    query_dtype = query_states.dtype
    rand_dtype = rand_prj.dtype

    if query_dtype == torch.half and rand_dtype == torch.half:
        return cuda_qjl_gqa_score.qjl_gqa_score_cuda_half_half(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    elif query_dtype == torch.half and rand_dtype == torch.float:
        return cuda_qjl_gqa_score.qjl_gqa_score_cuda_half_float(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    elif query_dtype == torch.float and rand_dtype == torch.float:
        return cuda_qjl_gqa_score.qjl_gqa_score_cuda_float_float(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    elif query_dtype == torch.bfloat16 and rand_dtype == torch.bfloat16:
        return cuda_qjl_gqa_score.qjl_gqa_score_cuda_bf16_bf16(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    elif query_dtype == torch.bfloat16 and rand_dtype == torch.float:
        return cuda_qjl_gqa_score.qjl_gqa_score_cuda_bf16_float(
            key_quant, key_outlier_quant, key_norm, key_outlier_norm, outlier_indices, query_sketch, query_states, rand_prj)
    else:
        raise TypeError("Unsupported data types for QJL GQA score calculation")

