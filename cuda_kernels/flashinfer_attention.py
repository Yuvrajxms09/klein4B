from __future__ import annotations

import math

import torch


FLASHINFER_ATTENTION_BACKEND = "flashinfer-single-prefill"
FLASHINFER_FP16_REDUCTION_ATTENTION_BACKEND = "flashinfer-single-prefill-fp16-reduction"
FLASHINFER_NVFP4_ATTENTION_BACKEND = "flashinfer-nvfp4-sm120"


@torch.library.custom_op("klein::flashinfer_single_prefill_attention", mutates_args=())
def _flashinfer_single_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    use_fp16_qk_reduction: bool,
) -> torch.Tensor:
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(
            f"FlashInfer Klein attention requires [1, S, H, D], got {tuple(query.shape)}"
        )
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("FlashInfer Klein attention requires identical Q/K/V shapes")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"FlashInfer Klein attention requires FP16/BF16, got {query.dtype}"
        )
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("FlashInfer Klein attention requires CUDA tensors")
    if (
        not query.is_contiguous()
        or not key.is_contiguous()
        or not value.is_contiguous()
    ):
        raise ValueError("FlashInfer Klein attention requires contiguous Q/K/V")

    from flashinfer import single_prefill_with_kv_cache

    return single_prefill_with_kv_cache(
        query[0],
        key[0],
        value[0],
        causal=False,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=use_fp16_qk_reduction,
        sm_scale=1.0 / math.sqrt(query.shape[-1]),
        backend="auto",
    ).unsqueeze(0)


@_flashinfer_single_prefill_attention.register_fake
def _flashinfer_single_prefill_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    use_fp16_qk_reduction: bool,
) -> torch.Tensor:
    del key, value, use_fp16_qk_reduction
    return torch.empty_like(query)


def flashinfer_exact_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return _flashinfer_single_prefill_attention(query, key, value, False)


def flashinfer_fp16_reduction_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return _flashinfer_single_prefill_attention(query, key, value, True)


@torch.library.custom_op("klein::flashinfer_nvfp4_attention_sm120", mutates_args=())
def _flashinfer_nvfp4_attention_sm120(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(
            f"FlashInfer NVFP4 attention requires [1, S, H, D], got {tuple(query.shape)}"
        )
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("FlashInfer NVFP4 attention requires identical Q/K/V shapes")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"FlashInfer NVFP4 attention requires FP16/BF16, got {query.dtype}"
        )
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("FlashInfer NVFP4 attention requires CUDA tensors")
    if torch.cuda.get_device_capability(query.device) != (12, 0):
        raise RuntimeError("FlashInfer NVFP4 attention requires SM120")

    from flashinfer.nvfp4_attention_sm120 import (
        nvfp4_attention_sm120_fwd,
        nvfp4_attention_sm120_quantize_qkv,
    )

    # FlashInfer's SM120 operator uses contiguous [B, H, S, D]. Include these
    # layout conversions in the candidate benchmark rather than hiding their cost.
    sequence = query.shape[1]
    query_bhsd = query.permute(0, 2, 1, 3).contiguous()
    key_bhsd = key.permute(0, 2, 1, 3).contiguous()
    value_bhsd = value.permute(0, 2, 1, 3).contiguous()
    packed = nvfp4_attention_sm120_quantize_qkv(
        query_bhsd,
        key_bhsd,
        value_bhsd,
        per_block_mean=True,
    )
    output, _ = nvfp4_attention_sm120_fwd(
        *packed,
        sm_scale=1.0 / math.sqrt(query.shape[-1]),
        causal=False,
        per_block_mean=True,
        out_dtype=query.dtype,
    )
    output = output[:, :, :sequence, :]
    return output.permute(0, 2, 1, 3).contiguous()


@_flashinfer_nvfp4_attention_sm120.register_fake
def _flashinfer_nvfp4_attention_sm120_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    del key, value
    return torch.empty_like(query)


def flashinfer_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return _flashinfer_nvfp4_attention_sm120(query, key, value)
