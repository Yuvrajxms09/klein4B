from __future__ import annotations

import torch


@torch.library.custom_op("klein::flashinfer_cudnn_fp4_mm", mutates_args=())
def flashinfer_cudnn_fp4_mm(
    activation_qdata: torch.Tensor,
    weight_qdata_t: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales_t: torch.Tensor,
    output_scale: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    """Graph-visible FlashInfer cuDNN NVFP4 GEMM using TorchAO-compatible packing."""
    from flashinfer import mm_fp4

    try:
        return mm_fp4(
            activation_qdata,
            weight_qdata_t,
            activation_scales,
            weight_scales_t,
            output_scale,
            out_dtype,
            backend="cudnn",
        )
    except Exception as exc:
        raise RuntimeError(
            "FlashInfer cuDNN NVFP4 GEMM failed "
            f"a={tuple(activation_qdata.shape)} b={tuple(weight_qdata_t.shape)} "
            f"a_scales={tuple(activation_scales.shape)} "
            f"b_scales={tuple(weight_scales_t.shape)} "
            f"error={type(exc).__name__}: {exc}"
        ) from None


@flashinfer_cudnn_fp4_mm.register_fake
def _flashinfer_cudnn_fp4_mm_fake(
    activation_qdata: torch.Tensor,
    weight_qdata_t: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales_t: torch.Tensor,
    output_scale: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    del weight_qdata_t, activation_scales, weight_scales_t, output_scale
    return activation_qdata.new_empty(
        (activation_qdata.shape[0], out_features), dtype=out_dtype
    )
