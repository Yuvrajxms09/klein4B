from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


logger = logging.getLogger(__name__)

# Triton accumulates RMSNorm and SwiGLU intermediates in FP32 before the final
# BF16 store. Diffusers can round at different points, so one or two BF16 ULPs
# are expected even when the layout and operation are correct.
BF16_VALIDATION_MAX_ABS = 0.05
BF16_VALIDATION_RELATIVE_L2 = 0.005


def _dispatch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    backend: Any,
    attention_mask: torch.Tensor | None,
    parallel_config: Any,
) -> torch.Tensor:
    from cuda_kernels.flashinfer_attention import (
        FLASHINFER_ATTENTION_BACKEND,
        FLASHINFER_FP16_REDUCTION_ATTENTION_BACKEND,
        FLASHINFER_NVFP4_ATTENTION_BACKEND,
    )

    if backend in {
        FLASHINFER_ATTENTION_BACKEND,
        FLASHINFER_FP16_REDUCTION_ATTENTION_BACKEND,
        FLASHINFER_NVFP4_ATTENTION_BACKEND,
    }:
        if attention_mask is not None or parallel_config is not None:
            raise ValueError(
                "FlashInfer Klein attention does not support masks or context parallelism"
            )
        from cuda_kernels.flashinfer_attention import (
            flashinfer_exact_attention,
            flashinfer_fp16_reduction_attention,
            flashinfer_nvfp4_attention,
        )

        attention_fn = {
            FLASHINFER_ATTENTION_BACKEND: flashinfer_exact_attention,
            FLASHINFER_FP16_REDUCTION_ATTENTION_BACKEND: flashinfer_fp16_reduction_attention,
            FLASHINFER_NVFP4_ATTENTION_BACKEND: flashinfer_nvfp4_attention,
        }[backend]
        return attention_fn(query, key, value)

    from diffusers.models.attention_dispatch import dispatch_attention_fn

    return dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        backend=backend,
        parallel_config=parallel_config,
    )


@triton.jit
def _fused_attention_swiglu_pack_kernel(
    attention_ptr,
    projected_ptr,
    output_ptr,
    projection_stride,
    output_stride,
    HIDDEN_SIZE: tl.constexpr,
    MLP_HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    attention_mask = offsets < HIDDEN_SIZE
    attention_values = tl.load(
        attention_ptr + row * HIDDEN_SIZE + offsets,
        mask=attention_mask,
        other=0.0,
    )
    tl.store(
        output_ptr + row * output_stride + offsets,
        attention_values,
        mask=attention_mask,
    )

    mlp_mask = offsets < MLP_HIDDEN_SIZE
    mlp_base = row * projection_stride + 3 * HIDDEN_SIZE
    gate = tl.load(
        projected_ptr + mlp_base + offsets,
        mask=mlp_mask,
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        projected_ptr + mlp_base + MLP_HIDDEN_SIZE + offsets,
        mask=mlp_mask,
        other=0.0,
    ).to(tl.float32)
    swiglu = gate * tl.sigmoid(gate) * value
    tl.store(
        output_ptr + row * output_stride + HIDDEN_SIZE + offsets,
        swiglu,
        mask=mlp_mask,
    )


@triton.jit
def _fused_single_qkv_rmsnorm_rope_pack_kernel(
    projected_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    sequence,
    heads,
    projection_stride,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    sequence_pos = row % sequence
    head_offset = head * HEAD_DIM
    source_base = row * projection_stride + head_offset
    half_offsets = tl.arange(0, HALF_DIM)
    even_offsets = half_offsets * 2
    odd_offsets = even_offsets + 1
    hidden_size = heads * HEAD_DIM

    q_even = tl.load(projected_ptr + source_base + even_offsets).to(tl.float32)
    q_odd = tl.load(projected_ptr + source_base + odd_offsets).to(tl.float32)
    k_even = tl.load(projected_ptr + source_base + hidden_size + even_offsets).to(
        tl.float32
    )
    k_odd = tl.load(projected_ptr + source_base + hidden_size + odd_offsets).to(
        tl.float32
    )
    q_rstd = tl.rsqrt(
        (tl.sum(q_even * q_even, axis=0) + tl.sum(q_odd * q_odd, axis=0)) / HEAD_DIM
        + eps
    )
    k_rstd = tl.rsqrt(
        (tl.sum(k_even * k_even, axis=0) + tl.sum(k_odd * k_odd, axis=0)) / HEAD_DIM
        + eps
    )
    q_even *= q_rstd * tl.load(q_weight_ptr + even_offsets).to(tl.float32)
    q_odd *= q_rstd * tl.load(q_weight_ptr + odd_offsets).to(tl.float32)
    k_even *= k_rstd * tl.load(k_weight_ptr + even_offsets).to(tl.float32)
    k_odd *= k_rstd * tl.load(k_weight_ptr + odd_offsets).to(tl.float32)

    rope_base = sequence_pos * HEAD_DIM
    cos_values = tl.load(cos_ptr + rope_base + even_offsets).to(tl.float32)
    sin_values = tl.load(sin_ptr + rope_base + even_offsets).to(tl.float32)
    output_base = (row * heads + head) * HEAD_DIM
    tl.store(
        query_ptr + output_base + even_offsets, q_even * cos_values - q_odd * sin_values
    )
    tl.store(
        query_ptr + output_base + odd_offsets, q_odd * cos_values + q_even * sin_values
    )
    tl.store(
        key_ptr + output_base + even_offsets, k_even * cos_values - k_odd * sin_values
    )
    tl.store(
        key_ptr + output_base + odd_offsets, k_odd * cos_values + k_even * sin_values
    )

    value_offsets = tl.arange(0, HEAD_DIM)
    values = tl.load(projected_ptr + source_base + 2 * hidden_size + value_offsets)
    tl.store(value_ptr + output_base + value_offsets, values)


@triton.jit
def _fused_joint_qkv_rmsnorm_rope_pack_kernel(
    image_qkv_ptr,
    text_qkv_ptr,
    joint_q_ptr,
    joint_k_ptr,
    joint_v_ptr,
    image_q_weight_ptr,
    image_k_weight_ptr,
    text_q_weight_ptr,
    text_k_weight_ptr,
    cos_ptr,
    sin_ptr,
    image_seq,
    text_seq,
    joint_seq,
    heads,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    eps: tl.constexpr,
):
    joint_row = tl.program_id(0)
    head = tl.program_id(1)
    batch = joint_row // joint_seq
    joint_pos = joint_row % joint_seq
    is_text = joint_pos < text_seq

    text_row = batch * text_seq + joint_pos
    image_pos = joint_pos - text_seq
    image_row = batch * image_seq + image_pos
    projection_width = 3 * heads * HEAD_DIM
    head_offset = head * HEAD_DIM
    half_offsets = tl.arange(0, HALF_DIM)
    even_offsets = half_offsets * 2
    odd_offsets = even_offsets + 1

    text_base = text_row * projection_width + head_offset
    image_base = image_row * projection_width + head_offset

    q_text_even = tl.load(
        text_qkv_ptr + text_base + even_offsets, mask=is_text, other=0.0
    )
    q_text_odd = tl.load(
        text_qkv_ptr + text_base + odd_offsets, mask=is_text, other=0.0
    )
    q_image_even = tl.load(
        image_qkv_ptr + image_base + even_offsets, mask=~is_text, other=0.0
    )
    q_image_odd = tl.load(
        image_qkv_ptr + image_base + odd_offsets, mask=~is_text, other=0.0
    )
    q_even = tl.where(is_text, q_text_even, q_image_even).to(tl.float32)
    q_odd = tl.where(is_text, q_text_odd, q_image_odd).to(tl.float32)

    k_projection_offset = heads * HEAD_DIM
    k_text_even = tl.load(
        text_qkv_ptr + text_base + k_projection_offset + even_offsets,
        mask=is_text,
        other=0.0,
    )
    k_text_odd = tl.load(
        text_qkv_ptr + text_base + k_projection_offset + odd_offsets,
        mask=is_text,
        other=0.0,
    )
    k_image_even = tl.load(
        image_qkv_ptr + image_base + k_projection_offset + even_offsets,
        mask=~is_text,
        other=0.0,
    )
    k_image_odd = tl.load(
        image_qkv_ptr + image_base + k_projection_offset + odd_offsets,
        mask=~is_text,
        other=0.0,
    )
    k_even = tl.where(is_text, k_text_even, k_image_even).to(tl.float32)
    k_odd = tl.where(is_text, k_text_odd, k_image_odd).to(tl.float32)

    q_variance = (
        tl.sum(q_even * q_even, axis=0) + tl.sum(q_odd * q_odd, axis=0)
    ) / HEAD_DIM
    k_variance = (
        tl.sum(k_even * k_even, axis=0) + tl.sum(k_odd * k_odd, axis=0)
    ) / HEAD_DIM
    q_rstd = tl.rsqrt(q_variance + eps)
    k_rstd = tl.rsqrt(k_variance + eps)

    text_q_weight_even = tl.load(
        text_q_weight_ptr + even_offsets, mask=is_text, other=0.0
    )
    text_q_weight_odd = tl.load(
        text_q_weight_ptr + odd_offsets, mask=is_text, other=0.0
    )
    image_q_weight_even = tl.load(
        image_q_weight_ptr + even_offsets, mask=~is_text, other=0.0
    )
    image_q_weight_odd = tl.load(
        image_q_weight_ptr + odd_offsets, mask=~is_text, other=0.0
    )
    text_k_weight_even = tl.load(
        text_k_weight_ptr + even_offsets, mask=is_text, other=0.0
    )
    text_k_weight_odd = tl.load(
        text_k_weight_ptr + odd_offsets, mask=is_text, other=0.0
    )
    image_k_weight_even = tl.load(
        image_k_weight_ptr + even_offsets, mask=~is_text, other=0.0
    )
    image_k_weight_odd = tl.load(
        image_k_weight_ptr + odd_offsets, mask=~is_text, other=0.0
    )

    q_weight_even = tl.where(is_text, text_q_weight_even, image_q_weight_even).to(
        tl.float32
    )
    q_weight_odd = tl.where(is_text, text_q_weight_odd, image_q_weight_odd).to(
        tl.float32
    )
    k_weight_even = tl.where(is_text, text_k_weight_even, image_k_weight_even).to(
        tl.float32
    )
    k_weight_odd = tl.where(is_text, text_k_weight_odd, image_k_weight_odd).to(
        tl.float32
    )

    q_even = q_even * q_rstd * q_weight_even
    q_odd = q_odd * q_rstd * q_weight_odd
    k_even = k_even * k_rstd * k_weight_even
    k_odd = k_odd * k_rstd * k_weight_odd

    rope_base = joint_pos * HEAD_DIM
    cos_values = tl.load(cos_ptr + rope_base + even_offsets).to(tl.float32)
    sin_values = tl.load(sin_ptr + rope_base + even_offsets).to(tl.float32)
    q_rotated_even = q_even * cos_values - q_odd * sin_values
    q_rotated_odd = q_odd * cos_values + q_even * sin_values
    k_rotated_even = k_even * cos_values - k_odd * sin_values
    k_rotated_odd = k_odd * cos_values + k_even * sin_values

    output_base = (joint_row * heads + head) * HEAD_DIM
    tl.store(joint_q_ptr + output_base + even_offsets, q_rotated_even)
    tl.store(joint_q_ptr + output_base + odd_offsets, q_rotated_odd)
    tl.store(joint_k_ptr + output_base + even_offsets, k_rotated_even)
    tl.store(joint_k_ptr + output_base + odd_offsets, k_rotated_odd)

    value_offsets = tl.arange(0, HEAD_DIM)
    value_projection_offset = 2 * heads * HEAD_DIM
    text_values = tl.load(
        text_qkv_ptr + text_base + value_projection_offset + value_offsets,
        mask=is_text,
        other=0.0,
    )
    image_values = tl.load(
        image_qkv_ptr + image_base + value_projection_offset + value_offsets,
        mask=~is_text,
        other=0.0,
    )
    value_values = tl.where(is_text, text_values, image_values)
    tl.store(joint_v_ptr + output_base + value_offsets, value_values)


@triton_op("klein::fused_joint_qkv_rmsnorm_rope_pack", mutates_args={})
def fused_joint_qkv_rmsnorm_rope_pack(
    image_qkv: torch.Tensor,
    text_qkv: torch.Tensor,
    image_q_weight: torch.Tensor,
    image_k_weight: torch.Tensor,
    text_q_weight: torch.Tensor,
    text_k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if image_qkv.ndim != 3 or text_qkv.ndim != 3:
        raise ValueError(
            "image_qkv and text_qkv must have shape [batch, sequence, 3 * hidden]"
        )
    if image_qkv.shape[0] != text_qkv.shape[0]:
        raise ValueError("image and text QKV batch sizes must match")
    if image_qkv.shape[-1] != text_qkv.shape[-1]:
        raise ValueError("image and text QKV projection widths must match")
    if image_qkv.dtype != torch.bfloat16 or text_qkv.dtype != torch.bfloat16:
        raise TypeError(
            "fused Klein QKV packing currently requires BF16 projection outputs"
        )
    if not image_qkv.is_contiguous() or not text_qkv.is_contiguous():
        raise ValueError(
            "fused Klein QKV packing requires contiguous projection outputs"
        )

    head_dim = int(image_q_weight.numel())
    if head_dim != 128:
        raise ValueError(
            f"fused Klein QKV packing is specialized for head_dim=128, got {head_dim}"
        )
    hidden_size, remainder = divmod(image_qkv.shape[-1], 3)
    heads, remainder = divmod(hidden_size, head_dim)
    if remainder:
        raise ValueError(
            "QKV projection width is incompatible with the RMSNorm head dimension"
        )

    batch = image_qkv.shape[0]
    image_seq = image_qkv.shape[1]
    text_seq = text_qkv.shape[1]
    joint_seq = image_seq + text_seq
    expected_rope_shape = (joint_seq, head_dim)
    if (
        tuple(cos.shape) != expected_rope_shape
        or tuple(sin.shape) != expected_rope_shape
    ):
        raise ValueError(
            f"expected RoPE tensors with shape {expected_rope_shape}, "
            f"got cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
        )

    output_shape = (batch, joint_seq, heads, head_dim)
    joint_q = torch.empty(output_shape, device=image_qkv.device, dtype=image_qkv.dtype)
    joint_k = torch.empty_like(joint_q)
    joint_v = torch.empty_like(joint_q)
    wrap_triton(_fused_joint_qkv_rmsnorm_rope_pack_kernel)[(batch * joint_seq, heads)](
        image_qkv,
        text_qkv,
        joint_q,
        joint_k,
        joint_v,
        image_q_weight,
        image_k_weight,
        text_q_weight,
        text_k_weight,
        cos,
        sin,
        image_seq,
        text_seq,
        joint_seq,
        heads,
        HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        eps=eps,
        num_warps=1,
    )
    return joint_q, joint_k, joint_v


@triton_op("klein::fused_single_qkv_rmsnorm_rope_pack", mutates_args={})
def fused_single_qkv_rmsnorm_rope_pack(
    projected: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if projected.ndim != 3 or projected.dtype != torch.bfloat16:
        raise ValueError(
            "single-block projection must be a BF16 [batch, sequence, channels] tensor"
        )
    if not projected.is_contiguous():
        raise ValueError(
            "single-block fused QKV+MLP projection output must be contiguous"
        )
    head_dim = int(q_weight.numel())
    if head_dim != 128 or k_weight.numel() != head_dim:
        raise ValueError(
            "single-block fused QKV packing is specialized for head_dim=128"
        )
    hidden_size = heads * head_dim
    if projected.shape[-1] < 3 * hidden_size:
        raise ValueError(
            "single-block projection does not contain a complete QKV prefix"
        )
    batch, sequence = projected.shape[:2]
    if tuple(cos.shape) != (sequence, head_dim) or tuple(sin.shape) != (
        sequence,
        head_dim,
    ):
        raise ValueError(
            "single-block RoPE tensors do not match the projected sequence"
        )

    output_shape = (batch, sequence, heads, head_dim)
    query = torch.empty(output_shape, device=projected.device, dtype=projected.dtype)
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    wrap_triton(_fused_single_qkv_rmsnorm_rope_pack_kernel)[(batch * sequence, heads)](
        projected,
        query,
        key,
        value,
        q_weight,
        k_weight,
        cos,
        sin,
        sequence,
        heads,
        projected.stride(1),
        HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        eps=eps,
        num_warps=1,
    )
    return query, key, value


@triton_op("klein::fused_attention_swiglu_pack", mutates_args={})
def fused_attention_swiglu_pack(
    attention_output: torch.Tensor,
    projected: torch.Tensor,
    hidden_size: int,
    mlp_hidden_size: int,
) -> torch.Tensor:
    """Pack flattened attention and SwiGLU output for the single-block output GEMM."""
    if attention_output.ndim != 4 or projected.ndim != 3:
        raise ValueError("expected attention [B, S, H, D] and projection [B, S, C]")
    if attention_output.dtype != torch.bfloat16 or projected.dtype != torch.bfloat16:
        raise TypeError("fused attention/SwiGLU packing requires BF16 tensors")
    if attention_output.shape[:2] != projected.shape[:2]:
        raise ValueError(
            "attention and projection batch/sequence dimensions must match"
        )
    if attention_output.shape[2] * attention_output.shape[3] != hidden_size:
        raise ValueError("attention head layout does not match hidden_size")
    expected_projection = 3 * hidden_size + 2 * mlp_hidden_size
    if projected.shape[-1] != expected_projection:
        raise ValueError(
            f"expected projection width {expected_projection}, got {projected.shape[-1]}"
        )
    if not attention_output.is_contiguous() or not projected.is_contiguous():
        raise ValueError("fused attention/SwiGLU packing requires contiguous tensors")

    batch, sequence = projected.shape[:2]
    output = torch.empty(
        (batch, sequence, hidden_size + mlp_hidden_size),
        device=projected.device,
        dtype=projected.dtype,
    )
    rows = batch * sequence
    grid = (rows, max(triton.cdiv(hidden_size, 256), triton.cdiv(mlp_hidden_size, 256)))
    wrap_triton(_fused_attention_swiglu_pack_kernel)[grid](
        attention_output,
        projected,
        output,
        projected.stride(1),
        output.stride(1),
        HIDDEN_SIZE=hidden_size,
        MLP_HIDDEN_SIZE=mlp_hidden_size,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


class KleinFusedQKVPackingProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, fallback: Any):
        self.fallback = fallback
        self._attention_backend = getattr(fallback, "_attention_backend", None)
        self._parallel_config = getattr(fallback, "_parallel_config", None)
        self._fallback_logged = False

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if (
            encoder_hidden_states is None
            or image_rotary_emb is None
            or not getattr(attn, "fused_projections", False)
            or not hasattr(attn, "to_qkv")
            or not hasattr(attn, "to_added_qkv")
        ):
            if not self._fallback_logged:
                logger.warning(
                    "Klein fused QKV packing fell back because the "
                    "double-stream contract was not met"
                )
                self._fallback_logged = True
            return self.fallback(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )

        image_qkv = attn.to_qkv(hidden_states)
        text_qkv = attn.to_added_qkv(encoder_hidden_states)
        query, key, value = fused_joint_qkv_rmsnorm_rope_pack(
            image_qkv,
            text_qkv,
            attn.norm_q.weight,
            attn.norm_k.weight,
            attn.norm_added_q.weight,
            attn.norm_added_k.weight,
            image_rotary_emb[0],
            image_rotary_emb[1],
            float(attn.norm_q.eps),
        )
        joint_output = _dispatch_attention(
            query,
            key,
            value,
            backend=self._attention_backend,
            attention_mask=attention_mask,
            parallel_config=self._parallel_config,
        ).flatten(2, 3)
        joint_output = joint_output.to(query.dtype)
        text_output, image_output = joint_output.split_with_sizes(
            [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
        )
        image_output = attn.to_out[1](attn.to_out[0](image_output))
        text_output = attn.to_add_out(text_output)
        return image_output, text_output


class KleinFusedSingleQKVPackingProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, fallback: Any):
        self.fallback = fallback
        self._attention_backend = getattr(fallback, "_attention_backend", None)
        self._parallel_config = getattr(fallback, "_parallel_config", None)

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if image_rotary_emb is None:
            return self.fallback(attn, hidden_states, attention_mask, image_rotary_emb)

        projected = attn.to_qkv_mlp_proj(hidden_states)
        query, key, value = fused_single_qkv_rmsnorm_rope_pack(
            projected,
            attn.norm_q.weight,
            attn.norm_k.weight,
            image_rotary_emb[0],
            image_rotary_emb[1],
            int(attn.heads),
            float(attn.norm_q.eps),
        )
        attention_output = _dispatch_attention(
            query,
            key,
            value,
            backend=self._attention_backend,
            attention_mask=attention_mask,
            parallel_config=self._parallel_config,
        )
        attention_output = attention_output.to(query.dtype)
        packed = fused_attention_swiglu_pack(
            attention_output,
            projected,
            int(attn.inner_dim),
            int(attn.mlp_hidden_dim),
        )
        return attn.to_out(packed)


def _validate_fused_pack(module: Any) -> dict[str, float]:
    from diffusers.models.embeddings import apply_rotary_emb

    batch, image_seq, text_seq = 1, 32, 16
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    generator = torch.Generator(device=module.norm_q.weight.device).manual_seed(0)
    image_qkv = torch.randn(
        (batch, image_seq, 3 * heads * head_dim),
        generator=generator,
        device=module.norm_q.weight.device,
        dtype=torch.bfloat16,
    )
    text_qkv = torch.randn(
        (batch, text_seq, 3 * heads * head_dim),
        generator=generator,
        device=module.norm_q.weight.device,
        dtype=torch.bfloat16,
    )
    angles = torch.randn(
        (image_seq + text_seq, head_dim // 2),
        generator=generator,
        device=module.norm_q.weight.device,
        dtype=torch.float32,
    )
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)

    def reference_stream(qkv, q_weight, k_weight, rope):
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (heads, head_dim))
        key = key.unflatten(-1, (heads, head_dim))
        value = value.unflatten(-1, (heads, head_dim))
        query = F.rms_norm(query, (head_dim,), q_weight, float(module.norm_q.eps))
        key = F.rms_norm(key, (head_dim,), k_weight, float(module.norm_q.eps))
        query = apply_rotary_emb(query, rope, sequence_dim=1)
        key = apply_rotary_emb(key, rope, sequence_dim=1)
        return query, key, value

    text_reference = reference_stream(
        text_qkv,
        module.norm_added_q.weight,
        module.norm_added_k.weight,
        (cos[:text_seq], sin[:text_seq]),
    )
    image_reference = reference_stream(
        image_qkv,
        module.norm_q.weight,
        module.norm_k.weight,
        (cos[text_seq:], sin[text_seq:]),
    )
    reference = tuple(
        torch.cat((text, image), dim=1)
        for text, image in zip(text_reference, image_reference)
    )
    candidate = fused_joint_qkv_rmsnorm_rope_pack(
        image_qkv,
        text_qkv,
        module.norm_q.weight,
        module.norm_k.weight,
        module.norm_added_q.weight,
        module.norm_added_k.weight,
        cos,
        sin,
        float(module.norm_q.eps),
    )
    max_abs = max(
        float((actual.float() - expected.float()).abs().max())
        for actual, expected in zip(candidate, reference)
    )
    relative_l2 = max(
        float(
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-12)
        )
        for actual, expected in zip(candidate, reference)
    )
    report = {"max_abs": max_abs, "relative_l2": relative_l2}
    if max_abs > BF16_VALIDATION_MAX_ABS or relative_l2 > BF16_VALIDATION_RELATIVE_L2:
        raise RuntimeError(
            f"Klein fused QKV packing numerical validation failed: {report}"
        )
    return report


def _validate_single_pack(module: Any) -> dict[str, float]:
    from diffusers.models.embeddings import apply_rotary_emb

    batch, sequence = 1, 32
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    generator = torch.Generator(device=module.norm_q.weight.device).manual_seed(1)
    projected = torch.randn(
        (batch, sequence, module.to_qkv_mlp_proj.out_features),
        generator=generator,
        device=module.norm_q.weight.device,
        dtype=torch.bfloat16,
    )
    angles = torch.randn(
        (sequence, head_dim // 2),
        generator=generator,
        device=module.norm_q.weight.device,
        dtype=torch.float32,
    )
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)

    qkv = projected[..., : 3 * module.inner_dim]
    query, key, value = qkv.chunk(3, dim=-1)
    query = query.unflatten(-1, (heads, head_dim))
    key = key.unflatten(-1, (heads, head_dim))
    value = value.unflatten(-1, (heads, head_dim))
    query = F.rms_norm(
        query, (head_dim,), module.norm_q.weight, float(module.norm_q.eps)
    )
    key = F.rms_norm(key, (head_dim,), module.norm_k.weight, float(module.norm_k.eps))
    reference = (
        apply_rotary_emb(query, (cos, sin), sequence_dim=1),
        apply_rotary_emb(key, (cos, sin), sequence_dim=1),
        value,
    )
    candidate = fused_single_qkv_rmsnorm_rope_pack(
        projected,
        module.norm_q.weight,
        module.norm_k.weight,
        cos,
        sin,
        heads,
        float(module.norm_q.eps),
    )
    max_abs = max(
        float((actual.float() - expected.float()).abs().max())
        for actual, expected in zip(candidate, reference)
    )
    relative_l2 = max(
        float(
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-12)
        )
        for actual, expected in zip(candidate, reference)
    )
    attention = torch.randn(
        (batch, sequence, heads, head_dim),
        generator=generator,
        device=module.norm_q.weight.device,
        dtype=torch.bfloat16,
    )
    mlp_offset = 3 * module.inner_dim
    mlp_width = module.mlp_hidden_dim * module.mlp_mult_factor
    mlp = projected.narrow(-1, mlp_offset, mlp_width)
    packed_reference = torch.cat(
        (attention.flatten(2, 3), module.mlp_act_fn(mlp)), dim=-1
    )
    packed_candidate = fused_attention_swiglu_pack(
        attention, projected, int(module.inner_dim), int(module.mlp_hidden_dim)
    )
    packed_max_abs = float(
        (packed_candidate.float() - packed_reference.float()).abs().max()
    )
    packed_relative_l2 = float(
        (packed_candidate.float() - packed_reference.float()).norm()
        / packed_reference.float().norm().clamp_min(1e-12)
    )
    report = {
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "post_attention_max_abs": packed_max_abs,
        "post_attention_relative_l2": packed_relative_l2,
    }
    if (
        max_abs > BF16_VALIDATION_MAX_ABS
        or relative_l2 > BF16_VALIDATION_RELATIVE_L2
        or packed_max_abs > BF16_VALIDATION_MAX_ABS
        or packed_relative_l2 > BF16_VALIDATION_RELATIVE_L2
    ):
        raise RuntimeError(
            "Klein fused single-block QKV packing numerical validation failed: "
            f"{report}"
        )
    return report


def install_fused_qkv_packing(
    transformer: Any, *, validate: bool = True
) -> dict[str, Any]:
    double_modules = [block.attn for block in transformer.transformer_blocks]
    single_modules = [block.attn for block in transformer.single_transformer_blocks]
    if len(double_modules) != 5 or len(single_modules) != 20:
        raise RuntimeError(
            "expected Klein 5+20 block topology, "
            f"got double={len(double_modules)} single={len(single_modules)}"
        )
    if any(
        not getattr(module, "fused_projections", False) for module in double_modules
    ):
        raise RuntimeError(
            "QKV projections must be fused before installing the "
            "Klein packing processor"
        )

    validation = (
        {
            "double": _validate_fused_pack(double_modules[0]),
            "single": _validate_single_pack(single_modules[0]),
        }
        if validate
        else None
    )
    for module in double_modules:
        module.set_processor(KleinFusedQKVPackingProcessor(module.processor))
    for module in single_modules:
        module.set_processor(KleinFusedSingleQKVPackingProcessor(module.processor))
    report = {
        "patched_double_blocks": len(double_modules),
        "patched_single_blocks": len(single_modules),
        "fused_post_attention_blocks": len(single_modules),
        "validation": validation,
    }
    transformer._klein_fused_qkv_packing_report = report
    logger.info("Qwen-style Klein fused QKV packing installed: %s", report)
    return report
