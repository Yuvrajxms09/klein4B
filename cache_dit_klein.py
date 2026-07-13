"""
Pipeline optimizations for FLUX.2 Klein 4B: cache-dit, attention backend, transformer compile.

- enable_cache_dit(pipe): DBCache on transformer (defaults match flux-stream-editor).
- apply_attention_backend(pipe, backend): Set transformer attention backend (works without cache-dit/compile).
- apply_transformer_compile(pipe, ...): torch.compile the transformer (works without cache-dit; if cache-dit
  is enabled, set_compile_configs() is called first).

All three are independent; use any subset. Biggest inference speedups: TAEF2 (VAE) > cache-dit > attention
backend > transformer compile (after warmup).
"""

from __future__ import annotations

import json
import logging
import statistics
import torch
from collections import Counter
from types import MethodType
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from PIL import Image

try:
    from einops import rearrange
except Exception:  # pragma: no cover - optional dependency in some envs
    rearrange = None


logger = logging.getLogger(__name__)


class _FixedKleinDenoiseLoop(torch.nn.Module):
    """Static four-step Klein denoiser suitable for one torch.compile/CUDA graph."""

    def __init__(self, transformer: torch.nn.Module, num_inference_steps: int):
        super().__init__()
        self.transformer = transformer
        self.num_inference_steps = int(num_inference_steps)

    def forward(
        self,
        latents: torch.Tensor,
        image_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler_dts: torch.Tensor,
        rotary_embeddings: tuple[Any, ...],
    ) -> torch.Tensor:
        latent_length = latents.shape[1]
        for step in range(self.num_inference_steps):
            latent_model_input = torch.cat((latents, image_latents), dim=1)
            timestep = timesteps[step].expand(latents.shape[0]).to(latents.dtype)
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                rotary_embeddings=rotary_embeddings,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :latent_length]
            latents = (latents.to(torch.float32) + scheduler_dts[step] * noise_pred.to(torch.float32)).to(
                noise_pred.dtype
            )
        return latents


def enable_compiled_denoise_loop(
    pipe: Any,
    *,
    num_inference_steps: int = 4,
    mode: str = "max-autotune",
) -> torch.nn.Module:
    """Compile the complete fixed-step denoiser instead of compiling each transformer call separately."""
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if getattr(pipe, "_cache_dit_mod", None) is not None:
        raise ValueError("whole-loop compile and Cache-DiT are separate benchmark tracks")
    loop = _FixedKleinDenoiseLoop(pipe.transformer, num_inference_steps).eval()
    compiled = torch.compile(loop, mode=mode, fullgraph=False, dynamic=False)
    pipe._compiled_denoise_loop = compiled
    pipe._compiled_denoise_loop_steps = int(num_inference_steps)
    logger.info("compiled resident Klein denoise loop steps=%d mode=%s", num_inference_steps, mode)
    return compiled


def latent_token_count_for_resolution(pipe: Any, height: int, width: int) -> int:
    """
    Compute packed latent token length (H*W in packed latent space) used by the denoiser.
    """
    multiple_of = int(pipe.vae_scale_factor) * 2
    h = 2 * (int(height) // multiple_of)
    w = 2 * (int(width) // multiple_of)
    return (h // 2) * (w // 2)


def load_ported_cuda_kernels(module_name: str = "klein_cuda_ext") -> Any:
    """
    Load compiled CUDA kernel extension module and return torch.ops namespace.
    """
    __import__(module_name)
    ns = getattr(torch.ops, "klein_cuda", None)
    if ns is None:
        raise RuntimeError(f"Loaded {module_name} but torch.ops.klein_cuda is unavailable")
    return ns


def apply_flux2_transformer_klein_ops(transformer: Any, *, verbose: bool = False) -> Any:
    """
    Patch an in-memory Flux2 transformer instance to use the local CUDA-backed
    optimization helpers where possible.

    This is intentionally conservative: it only patches block methods when the
    expected Flux2 block attributes exist. It can be called immediately after
    loading the transformer and before torch.compile().
    """
    if rearrange is None:
        raise RuntimeError("einops is required for Flux2 transformer patching")

    ns = getattr(torch.ops, "klein_cuda", None)
    if ns is None:
        try:
            from cuda_kernels import load_compiled_extension

            load_compiled_extension()
            ns = getattr(torch.ops, "klein_cuda", None)
        except Exception as exc:
            raise RuntimeError("torch.ops.klein_cuda is unavailable; load the extension first") from exc
    if ns is None:
        raise RuntimeError("torch.ops.klein_cuda is unavailable; load the extension first")

    def _has_op(name: str) -> bool:
        return hasattr(ns, name)

    # Prefer flash/mem-efficient SDPA globally so the hot path doesn't need a
    # per-call context manager. This keeps the live block wrapper lighter.
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

    def _split_qkv_heads(qkv: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = qkv.shape
        head_dim = qkv.shape[-1] // (3 * num_heads)
        qkv = qkv.view(batch, sequence, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        if not qkv.is_contiguous():
            qkv = qkv.contiguous()
        return qkv[0], qkv[1], qkv[2]

    def _split_modulation(
        mod: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        parts = mod.chunk(6, dim=-1)
        first = parts[:3]
        second = parts[3:]
        if first[0].ndim == 2:
            first = tuple(t.unsqueeze(1) for t in first)
            second = tuple(t.unsqueeze(1) for t in second)
        return first, second

    def _apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            xq.is_cuda
            and xk.is_cuda
            and xq.dtype == torch.float32
            and xk.dtype == torch.float32
            and xq.ndim == 4
            and xq.shape == xk.shape
            and freqs_cis.ndim == 6
            and xq.shape[0] == 1
            and freqs_cis.shape[0] == 1
        ):
            q3 = xq.squeeze(0).permute(1, 0, 2).contiguous()
            k3 = xk.squeeze(0).permute(1, 0, 2).contiguous()
            cos = freqs_cis[0, 0, :, :, 0, 0].contiguous()
            sin = freqs_cis[0, 0, :, :, 1, 0].contiguous()
            q3 = ns.rope_2d_offset_(q3, cos, sin, 0, q3.shape[0])
            k3 = ns.rope_2d_offset_(k3, cos, sin, 0, k3.shape[0])
            return q3.permute(1, 0, 2).unsqueeze(0), k3.permute(1, 0, 2).unsqueeze(0)
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def _get_work_buffer(module: Any, name: str, shape: tuple[int, ...], ref: torch.Tensor) -> torch.Tensor:
        key = f"_klein_{name}"
        buf = getattr(module, key, None)
        if buf is None or tuple(buf.shape) != tuple(shape) or buf.device != ref.device or buf.dtype != ref.dtype:
            buf = torch.empty(shape, device=ref.device, dtype=ref.dtype)
            setattr(module, key, buf)
        return buf

    def _cat_into_buffer(parts: tuple[torch.Tensor, ...], module: Any, name: str, dim: int = -1) -> torch.Tensor:
        out_shape = list(parts[0].shape)
        out_shape[dim] = sum(p.shape[dim] for p in parts)
        out = _get_work_buffer(module, name, tuple(out_shape), parts[0])
        offset = 0
        for part in parts:
            width = part.shape[dim]
            out.narrow(dim, offset, width).copy_(part)
            offset += width
        return out

    def _packed_attention_call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
        if not _has_op("packed_attention_"):
            return None
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            return None
        if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
            return None
        if q.shape != k.shape or q.shape != v.shape:
            return None
        if q.ndim == 4 and q.shape[0] == 1:
            q = q.squeeze(0).permute(1, 0, 2).contiguous()
            k = k.squeeze(0).permute(1, 0, 2).contiguous()
            v = v.squeeze(0).permute(1, 0, 2).contiguous()
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            return None
        if q.shape[-1] > 128:
            return None
        return ns.packed_attention_(q, k, v, 1.0 / (q.shape[-1] ** 0.5))

    def _seq_major_from_batched_qkv(qkv: torch.Tensor) -> torch.Tensor:
        if qkv.ndim != 5 or qkv.shape[2] != 3:
            raise ValueError(f"expected [batch, seq, 3, heads, head_dim], got {tuple(qkv.shape)}")
        if qkv.shape[0] != 1:
            raise ValueError(f"packed fused attention only supports batch=1, got {qkv.shape[0]}")
        qkv = qkv.squeeze(0).permute(0, 2, 1, 3)
        if not qkv.is_contiguous():
            qkv = qkv.contiguous()
        return qkv

    def _seq_major_from_bhq(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [batch, heads, seq, head_dim], got {tuple(x.shape)}")
        if x.shape[0] != 1:
            raise ValueError(f"packed attention only supports batch=1, got {x.shape[0]}")
        x = x.squeeze(0).permute(1, 0, 2)
        if not x.is_contiguous():
            x = x.contiguous()
        return x

    def _bhq_from_seq_major(x: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
        return x.reshape(batch, seq, x.shape[1], x.shape[2]).contiguous()

    def _flatten_attention_output(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.reshape(1, x.shape[0], -1)
            return x.contiguous() if not x.is_contiguous() else x
        if x.ndim == 4:
            x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
            return x.contiguous() if not x.is_contiguous() else x
        raise ValueError(f"unexpected attention output rank: {x.ndim}")

    def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        packed = _packed_attention_call(q, k, v)
        if packed is not None:
            return packed
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    def _qk_norm(module: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            q.is_cuda
            and k.is_cuda
            and q.dtype == torch.float32
            and k.dtype == torch.float32
            and q.shape == k.shape
            and q.ndim == 4
        ):
            qn = q.permute(0, 2, 1, 3).reshape(-1, q.shape[1], q.shape[3]).contiguous()
            kn = k.permute(0, 2, 1, 3).reshape(-1, k.shape[1], k.shape[3]).contiguous()
            qw = module.query_norm.scale.to(dtype=torch.float32, copy=False)
            kw = module.key_norm.scale.to(dtype=torch.float32, copy=False)
            qn = ns.qk_rms_norm_(qn, kn, qw, kw)
            q = qn.reshape(q.shape[0], q.shape[2], q.shape[1], q.shape[3]).permute(0, 2, 1, 3)
            k = kn.reshape(k.shape[0], k.shape[2], k.shape[1], k.shape[3]).permute(0, 2, 1, 3)
            if q.dtype != v.dtype:
                q = q.to(v)
            if k.dtype != v.dtype:
                k = k.to(v)
            return q, k
        q = module.query_norm(q)
        k = module.key_norm(k)
        if q.dtype != v.dtype:
            q = q.to(v)
        if k.dtype != v.dtype:
            k = k.to(v)
        return q, k

    def _fused_qkv_rope_qk_norm(
        qkv: torch.Tensor,
        module: Any,
        pe: torch.Tensor,
        seq_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not hasattr(ns, "fused_qkv_rope_qk_norm_"):
            return None
        if not (
            qkv.is_cuda
            and qkv.dtype in (torch.float32, torch.bfloat16)
            and qkv.ndim == 5
            and qkv.shape[2] == 3
        ):
            return None
        try:
            qw = module.query_norm.scale.to(dtype=torch.float32, copy=False)
            kw = module.key_norm.scale.to(dtype=torch.float32, copy=False)
            qkv3 = _seq_major_from_batched_qkv(qkv)
            q, k, v = ns.fused_qkv_rope_qk_norm_(
                qkv3,
                qw,
                kw,
                pe[0, 0, :, :, 0, 0].contiguous(),
                pe[0, 0, :, :, 1, 0].contiguous(),
                seq_offset,
                qkv3.shape[0],
            )
            # The CUDA op works in sequence-major storage; expose the same
            # [B, H, L, D] contract as Diffusers' attention fallback so BF16
            # continues through exact SDPA instead of the FP32-only custom op.
            return (
                q.unsqueeze(0).permute(0, 2, 1, 3),
                k.unsqueeze(0).permute(0, 2, 1, 3),
                v.unsqueeze(0).permute(0, 2, 1, 3),
            )
        except Exception as exc:
            if not getattr(transformer, "_klein_qkv_fallback_logged", False):
                logger.warning("fused QKV norm/RoPE op fell back to PyTorch attention path: %s", exc)
                setattr(transformer, "_klein_qkv_fallback_logged", True)
            return None

    def _fused_qkv_attention(
        qkv: torch.Tensor,
        module: Any,
        pe: torch.Tensor,
        seq_offset: int = 0,
    ) -> torch.Tensor | None:
        if not hasattr(ns, "fused_qkv_attention_"):
            return None
        if not (qkv.is_cuda and qkv.dtype == torch.float32 and qkv.ndim == 5 and qkv.shape[2] == 3):
            return None
        if qkv.shape[0] != 1:
            return None
        try:
            qw = module.query_norm.scale.to(dtype=torch.float32, copy=False)
            kw = module.key_norm.scale.to(dtype=torch.float32, copy=False)
            qkv3 = _seq_major_from_batched_qkv(qkv)
            return ns.fused_qkv_attention_(
                qkv3,
                qw,
                kw,
                pe[0, 0, :, :, 0, 0].contiguous(),
                pe[0, 0, :, :, 1, 0].contiguous(),
                seq_offset,
                qkv3.shape[0],
                1.0 / (qkv3.shape[-1] ** 0.5),
            )
        except Exception:
            return None

    def _maybe_run_fused_qkv_attention(
        qkv: torch.Tensor,
        module: Any,
        pe: torch.Tensor,
        seq_offset: int = 0,
    ) -> torch.Tensor | None:
        attn = _fused_qkv_attention(qkv, module, pe, seq_offset=seq_offset)
        if attn is not None:
            return _flatten_attention_output(attn)
        return None

    def _maybe_run_single_fused_qkv_attention(
        qkv: torch.Tensor,
        module: Any,
        pe: torch.Tensor,
        seq_offset: int = 0,
    ) -> torch.Tensor | None:
        if not (qkv.is_cuda and qkv.dtype == torch.float32 and qkv.ndim == 4 and qkv.shape[2] == 3):
            return None
        try:
            qkv3 = _seq_major_from_batched_qkv(qkv)
            attn = ns.fused_qkv_attention_(
                qkv3,
                module.query_norm.scale.to(dtype=torch.float32, copy=False),
                module.key_norm.scale.to(dtype=torch.float32, copy=False),
                pe[0, 0, :, :, 0, 0].contiguous(),
                pe[0, 0, :, :, 1, 0].contiguous(),
                seq_offset,
                qkv3.shape[0],
                1.0 / (qkv3.shape[-1] ** 0.5),
            )
            return _flatten_attention_output(attn)
        except Exception:
            return None

    def _silu_mul(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        if x1.is_cuda and x1.dtype == torch.float32 and x1.shape == x2.shape:
            return ns.silu_mul_(x1, x2)
        return torch.nn.functional.silu(x1) * x2

    def _adaln(module: Any, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if (
            _has_op("adaln_norm")
            and x.is_cuda
            and x.dtype == torch.float32
            and shift.is_cuda
            and scale.is_cuda
            and shift.dtype == torch.float32
            and scale.dtype == torch.float32
            and shift.ndim == 1
            and scale.ndim == 1
            and x.shape[-1] == shift.shape[0]
            and x.shape[-1] == scale.shape[0]
        ):
            return ns.adaln_norm(x, shift, scale, 1e-6)
        return module(x).mul_(1 + scale).add_(shift)

    def _single_forward(self, x: torch.Tensor, pe: torch.Tensor, mod: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = mod
        x_mod = _adaln(self.pre_norm, x, mod_shift, mod_scale)
        fused = self.linear1(x_mod)
        qkv = fused[..., : 3 * self.hidden_size]
        mlp = fused[..., 3 * self.hidden_size :]
        fused_qkv = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, qkv.shape[-1] // (3 * self.num_heads)).contiguous()
        attn = None
        if fused_qkv.shape[0] == 1:
            attn = _maybe_run_single_fused_qkv_attention(
                fused_qkv,
                self.norm,
                pe,
                seq_offset=0,
            )
        if attn is None:
            fused = _fused_qkv_rope_qk_norm(fused_qkv, self.norm, pe)
            if fused is not None:
                q, k, v = fused
            else:
                q, k, v = _split_qkv_heads(qkv, self.num_heads)
                q, k = _qk_norm(self.norm, q, k, v)
                q, k = _apply_rope(q, k, pe)
            attn = _packed_attention_call(q, k, v)
            if attn is None:
                attn = _attention(q, k, v)
            if attn is None:
                raise RuntimeError("attention path unexpectedly failed")
            attn = _flatten_attention_output(attn)
        mlp_act = _silu_mul(mlp)
        fused = _cat_into_buffer((attn, mlp_act), self, "single_concat", dim=-1)
        output = self.linear2(fused)
        return x + mod_gate * output

    def _single_forward_diffusers(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb_mod: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn = self.attn
        norm = self.norm
        if verbose:
            print("[klein] single block", type(self).__name__, "hidden", tuple(hidden_states.shape))
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = _cat_into_buffer((encoder_hidden_states, hidden_states), self, "single_hidden_states", dim=1)
        mod_shift, mod_scale, mod_gate = temb_mod.chunk(3, dim=-1)
        if mod_shift.ndim == 2:
            mod_shift = mod_shift.unsqueeze(1)
            mod_scale = mod_scale.unsqueeze(1)
            mod_gate = mod_gate.unsqueeze(1)
        norm_hidden_states = _adaln(norm, hidden_states, mod_shift, mod_scale)
        attn_output = attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        return hidden_states

    def _double_forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        pe: torch.Tensor,
        pe_ctx: torch.Tensor,
        mod_img: tuple[torch.Tensor, torch.Tensor],
        mod_txt: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt
        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        img_modulated = self.img_norm1(img)
        img_modulated = img_modulated.mul_(1 + img_mod1_scale).add_(img_mod1_shift)
        img_qkv = self.img_attn.qkv(img_modulated).view(
            img_modulated.shape[0], img_modulated.shape[1], 3, self.num_heads, self.img_attn.qkv.out_features // (3 * self.num_heads)
        ).contiguous()
        fused_img = _fused_qkv_rope_qk_norm(img_qkv, self.img_attn.norm, pe)
        if fused_img is None:
            img_q, img_k, img_v = _split_qkv_heads(img_qkv, self.num_heads)
            img_q, img_k = _qk_norm(self.img_attn.norm, img_q, img_k, img_v)
            img_q, img_k = _apply_rope(img_q, img_k, pe)
        else:
            img_q, img_k, img_v = fused_img

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = txt_modulated.mul_(1 + txt_mod1_scale).add_(txt_mod1_shift)
        txt_qkv = self.txt_attn.qkv(txt_modulated).view(
            txt_modulated.shape[0], txt_modulated.shape[1], 3, self.num_heads, self.txt_attn.qkv.out_features // (3 * self.num_heads)
        ).contiguous()
        fused_txt = _fused_qkv_rope_qk_norm(txt_qkv, self.txt_attn.norm, pe_ctx)
        if fused_txt is None:
            txt_q, txt_k, txt_v = _split_qkv_heads(txt_qkv, self.num_heads)
            txt_q, txt_k = _qk_norm(self.txt_attn.norm, txt_q, txt_k, txt_v)
            txt_q, txt_k = _apply_rope(txt_q, txt_k, pe_ctx)
        else:
            txt_q, txt_k, txt_v = fused_txt

        q = _cat_into_buffer((txt_q, img_q), self, "double_q", dim=2)
        k = _cat_into_buffer((txt_k, img_k), self, "double_k", dim=2)
        v = _cat_into_buffer((txt_v, img_v), self, "double_v", dim=2)
        attn = _flatten_attention_output(_attention(q, k, v))
        txt_attn = attn[:, : txt_q.shape[2]]
        img_attn = attn[:, txt_q.shape[2] :]

        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp(self.img_norm2(img).mul_(1 + img_mod2_scale).add_(img_mod2_shift))
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp(self.txt_norm2(txt).mul_(1 + txt_mod2_scale).add_(txt_mod2_shift))
        return img, txt

    def _double_forward_diffusers(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        if verbose:
            print("[klein] double block", type(self).__name__, "img", tuple(hidden_states.shape), "txt", tuple(encoder_hidden_states.shape))
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = _split_modulation(temb_mod_img)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = _split_modulation(temb_mod_txt)

        norm_hidden_states = _adaln(self.norm1, hidden_states, shift_msa, scale_msa)
        norm_encoder_hidden_states = _adaln(self.norm1_context, encoder_hidden_states, c_shift_msa, c_scale_msa)

        # Use packed attention when the Flux block exposes qkv modules; otherwise fall back.
        attn_hidden = None
        attn_context = None
        if all(
            hasattr(self, name)
            for name in ("qkv", "qkv_context", "proj", "img_attn", "txt_attn", "ff", "ff_context")
        ):
            try:
                qkv_hidden = self.qkv(norm_hidden_states)
                qkv_context = self.qkv_context(norm_encoder_hidden_states)
                q_hidden, k_hidden, v_hidden = _split_qkv_heads(qkv_hidden, self.num_heads)
                q_context, k_context, v_context = _split_qkv_heads(qkv_context, self.num_heads)
                q_hidden, k_hidden = _qk_norm(self.img_attn.norm, q_hidden, k_hidden, v_hidden)
                q_context, k_context = _qk_norm(self.txt_attn.norm, q_context, k_context, v_context)
                if image_rotary_emb is not None:
                    pe = image_rotary_emb[0]
                    q_hidden, k_hidden = _apply_rope(q_hidden, k_hidden, pe)
                    q_context, k_context = _apply_rope(q_context, k_context, pe)
                if hasattr(ns, "joint_packed_attention_"):
                    try:
                        q_hidden_3d = _seq_major_from_bhq(q_hidden)
                        k_hidden_3d = _seq_major_from_bhq(k_hidden)
                        v_hidden_3d = _seq_major_from_bhq(v_hidden)
                        q_context_3d = _seq_major_from_bhq(q_context)
                        k_context_3d = _seq_major_from_bhq(k_context)
                        v_context_3d = _seq_major_from_bhq(v_context)
                        if verbose:
                            logger.info(
                                "double_stream_joint_packed_attention seq_hidden=%s seq_context=%s heads=%s head_dim=%s",
                                q_hidden_3d.shape[0],
                                q_context_3d.shape[0],
                                q_hidden_3d.shape[1],
                                q_hidden_3d.shape[2],
                            )
                        attn_hidden, attn_context = ns.joint_packed_attention_(
                            q_hidden_3d,
                            k_hidden_3d,
                            v_hidden_3d,
                            q_context_3d,
                            k_context_3d,
                            v_context_3d,
                            1.0 / (q_hidden_3d.shape[-1] ** 0.5),
                        )
                        attn_hidden = _flatten_attention_output(attn_hidden)
                        attn_context = _flatten_attention_output(attn_context)
                    except Exception:
                        attn_hidden = None
                        attn_context = None
                if attn_hidden is None or attn_context is None:
                    hidden_attn = _packed_attention_call(q_hidden, k_hidden, v_hidden)
                    context_attn = _packed_attention_call(q_context, k_context, v_context)
                    if attn_hidden is None and hidden_attn is not None:
                        attn_hidden = _flatten_attention_output(hidden_attn)
                    if attn_context is None and context_attn is not None:
                        attn_context = _flatten_attention_output(context_attn)
                if attn_hidden is None or attn_context is None:
                    q = _cat_into_buffer((q_context, q_hidden), self, "double_q", dim=2)
                    k = _cat_into_buffer((k_context, k_hidden), self, "double_k", dim=2)
                    v = _cat_into_buffer((v_context, v_hidden), self, "double_v", dim=2)
                    if image_rotary_emb is not None:
                        pe = image_rotary_emb[0]
                        q, k = _apply_rope(q, k, pe)
                    attn = _flatten_attention_output(_attention(q, k, v))
                    attn_context = attn[:, : q_context.shape[2]]
                    attn_hidden = attn[:, q_context.shape[2] :]
            except Exception:
                attn_hidden = None
                attn_context = None

        if attn_hidden is None or attn_context is None:
            # Use the existing attention module, but keep the block math and projections in-place.
            attn_hidden, attn_context = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

        hidden_states = hidden_states + gate_msa * attn_hidden
        norm_hidden_states = _adaln(self.norm2, hidden_states, shift_mlp, scale_mlp)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * attn_context
        norm_encoder_hidden_states = _adaln(self.norm2_context, encoder_hidden_states, c_shift_mlp, c_scale_mlp)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states

    patched = 0
    for block in getattr(transformer, "transformer_blocks", []):
        if hasattr(block, "attn") and hasattr(block, "ff") and hasattr(block, "ff_context"):
            block.forward = MethodType(_double_forward_diffusers, block)
            patched += 1
    for block in getattr(transformer, "single_transformer_blocks", []):
        if hasattr(block, "attn") and hasattr(block, "norm"):
            block.forward = MethodType(_single_forward_diffusers, block)
            patched += 1

    if patched == 0:
        raise RuntimeError("No Flux2 blocks were patched; transformer structure did not match expectations")
    setattr(transformer, "_klein_cuda_ops_patched", True)
    logger.info(
        "Klein CUDA block patches active: double_blocks=%d single_blocks=%d fused_qkv_norm_rope=%s",
        len(getattr(transformer, "transformer_blocks", [])),
        len(getattr(transformer, "single_transformer_blocks", [])),
        _has_op("fused_qkv_rope_qk_norm_"),
    )
    return transformer


def enable_klein_c_cuda_backend(
    pipe: Any,
    *,
    model_dir: str,
    bridge_lib_path: str | None = None,
    enforce_single_ref: bool = False,
) -> None:
    """
    Use klein-cuda-c backend (Option A) for transformer denoising via ctypes bridge.

    This delegates denoising to klein-cuda-c runtime and keeps scheduler/VAE/text flow in Python.
    """
    if not hasattr(pipe, "set_cuda_denoiser"):
        raise AttributeError("Pipeline does not support custom CUDA denoiser registration")

    from klein_c_bridge import make_klein_c_denoiser

    denoiser = make_klein_c_denoiser(
        pipe,
        model_dir=model_dir,
        bridge_lib_path=bridge_lib_path,
        enforce_single_ref=enforce_single_ref,
    )
    pipe.set_cuda_denoiser(denoiser, name="klein-cuda-c-bridge")


def build_klein_c_full_backend(
    *,
    model_dir: str,
    lib_path: str | None = None,
    use_mmap: bool = True,
) -> Any:
    """
    Build full native klein-cuda-c backend for end-to-end C/CUDA img2img and multiref.
    """
    from klein_c_full_backend import KleinCFullBackend

    return KleinCFullBackend(model_dir=model_dir, lib_path=lib_path, use_mmap=use_mmap)


def enable_klein_c_full_backend(
    pipe: Any,
    *,
    model_dir: str,
    lib_path: str | None = None,
    use_mmap: bool = True,
) -> Any:
    """
    Attach the native klein-cuda-c backend to the pipeline for direct img2img execution.

    This bypasses the Python denoising loop entirely for image-conditioned generation.
    """
    backend = build_klein_c_full_backend(
        model_dir=model_dir,
        lib_path=lib_path,
        use_mmap=use_mmap,
    )
    pipe._klein_c_full_backend = backend
    return backend


class NunchakuKleinFullBackend:
    """
    Nunchaku-style end-to-end img2img backend for Klein4B.

    This keeps the denoising loop outside Diffusers' transformer call path and uses
    the Nunchaku Flux2 transformer implementation directly.
    """

    def __init__(self, *, model_dir: str, pipe: Any | None = None):
        self.model_dir = str(model_dir)
        self.pipe = pipe
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from nunchaku.models.transformers import NunchakuFlux2Transformer2DModel
        except Exception as exc:
            raise RuntimeError("nunchaku Flux2 transformer is not importable") from exc

        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"model_dir does not exist: {self.model_dir}")
        self._model = NunchakuFlux2Transformer2DModel.from_pretrained(
            model_path,
            device="cuda",
            torch_dtype=torch.float16,
            offload=False,
        )
        return self._model

    def _prepare_latents(self, image: torch.Tensor, generator: Optional[torch.Generator], dtype: torch.dtype):
        if self.pipe is None:
            raise RuntimeError("pipe is required for latent preprocessing")
        return self.pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=self.pipe.transformer.config.in_channels // 4,
            height=image.shape[-2],
            width=image.shape[-1],
            dtype=dtype,
            device=self.pipe._execution_device,
            generator=generator,
            latents=None,
        )

    def img2img(self, prompt: str, image: Image.Image, config: Any):
        if self.pipe is None:
            raise RuntimeError("NunchakuKleinFullBackend requires an attached pipe")
        model = self._load_model()
        pipe = self.pipe

        device = pipe._execution_device
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=getattr(config, "max_sequence_length", 512),
            text_encoder_out_layers=getattr(config, "text_encoder_out_layers", (9, 18, 27)),
        )
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = pipe.image_processor.preprocess(image, height=config.height, width=config.width, resize_mode="crop")
        image_latents, image_latent_ids = pipe.prepare_image_latents(
            images=[image],
            batch_size=1,
            generator=None if config.seed < 0 else torch.Generator(device=device).manual_seed(int(config.seed)),
            device=device,
            dtype=pipe.vae.dtype,
            non_blocking_h2d=True,
        )
        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=pipe.transformer.config.in_channels // 4,
            height=config.height,
            width=config.width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=None if config.seed < 0 else torch.Generator(device=device).manual_seed(int(config.seed)),
            latents=None,
        )
        latents = latents.to(model.dtype)
        image_latents = image_latents.to(model.dtype)
        latent_ids = latent_ids.to(device)
        image_latent_ids = image_latent_ids.to(device)
        timesteps, _ = pipe._get_timesteps_cached(latents.shape[1], int(config.num_steps), device)
        guidance = torch.tensor([float(config.guidance)], device=device, dtype=latents.dtype)
        for t in timesteps:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input, latent_image_ids = pipe._prepare_denoiser_inputs(
                latents=latents,
                image_latents=image_latents,
                latent_ids=latent_ids,
                image_latent_ids=image_latent_ids,
            )
            noise_pred = model(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        latent_height = int(config.height) // (pipe.vae_scale_factor * 2)
        latent_width = int(config.width) // (pipe.vae_scale_factor * 2)
        latents = pipe._unpack_latents_with_ids(
            latents,
            latent_ids,
            latent_height,
            latent_width,
        )
        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = pipe._unpatchify_latents(latents)
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        return image


def enable_nunchaku_full_backend(pipe: Any, *, model_dir: str) -> Any:
    backend = NunchakuKleinFullBackend(model_dir=model_dir, pipe=pipe)
    pipe._klein_c_full_backend = backend
    return backend


def enable_nunchaku_transformer(pipe: Any, *, model_dir: str) -> Any:
    """
    Replace the Diffusers transformer with a Nunchaku Flux2 transformer.

    This is the closest match to the nunchaku runtime style: the pipeline keeps
    its scheduler/VAE/prompt code, but the denoiser module itself becomes the
    Nunchaku implementation.
    """
    try:
        from nunchaku.models.transformers import NunchakuFlux2Transformer2DModel
    except Exception as exc:
        raise RuntimeError("nunchaku Flux2 transformer is not importable") from exc

    model = NunchakuFlux2Transformer2DModel.from_pretrained(
        model_dir,
        device="cuda",
        torch_dtype=torch.float16,
        offload=False,
    )
    pipe.transformer = model
    return model


def _parse_steps_mask(mask_text: str, expected_steps: int) -> list[int]:
    cleaned = mask_text.replace(",", "").replace(" ", "")
    if not cleaned:
        raise ValueError("steps mask cannot be empty")
    if any(ch not in ("0", "1") for ch in cleaned):
        raise ValueError(f"steps mask must only contain 0/1, got: {mask_text}")
    mask = [int(ch) for ch in cleaned]
    if len(mask) != expected_steps:
        raise ValueError(
            f"steps mask length mismatch: got {len(mask)}, expected {expected_steps} (num_inference_steps)",
        )
    return mask


def _default_steps_mask(num_inference_steps: int) -> str:
    """Same logic as flux-stream-editor build_default_config."""
    if num_inference_steps == 2:
        return "10"
    return "1" * num_inference_steps


def enable_cache_dit(
    pipe: Any,
    *,
    num_inference_steps: int = 4,
    steps_mask: str | None = None,
    cache_fn: int = 1,
    cache_bn: int = 0,
    residual_diff_threshold: float = 0.8,
    single_block_rdt_scale: float = 3.0,
    max_warmup_steps: int = 0,
    warmup_interval: int = 1,
    max_cached_steps: int = -1,
    max_continuous_cached_steps: int = -1,
    cache_enable_separate_cfg: bool = False,
    steps_computation_policy: str = "dynamic",
    enable_taylorseer: bool = True,
    taylorseer_order: int = 1,
    require_reuse_candidate: bool = False,
) -> dict[str, Any]:
    """
    Enable cache-dit (DBCache) on the pipeline's transformer. Call once after
    loading (e.g. after from_pretrained).

    Defaults match flux-stream-editor (FastFlux2Config): cache_fn=1, cache_bn=0,
    residual_diff_threshold=0.8, single_block_rdt_scale=3.0, max_warmup_steps=0.
    steps_mask: if None, uses "10" for 2 steps else "1"*num_inference_steps (same as
    flux-stream-editor build_default_config). Pipeline __call__ will call
    refresh_context(transformer, num_inference_steps=...) before each run.

    Requires: pip install cache-dit
    """
    try:
        import cache_dit as cache_dit_mod
        from cache_dit import (
            BlockAdapter,
            DBCacheConfig,
            ForwardPattern,
            ParamsModifier,
            TaylorSeerCalibratorConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "cache-dit is not available. Install it first, e.g. pip install cache-dit",
        ) from exc

    if steps_mask is None:
        steps_mask = _default_steps_mask(num_inference_steps)
    steps_computation_mask = _parse_steps_mask(steps_mask, num_inference_steps)
    if require_reuse_candidate and all(steps_computation_mask):
        raise ValueError("cache reuse benchmark cannot use an all-compute steps mask")

    cache_config = DBCacheConfig(
        Fn_compute_blocks=cache_fn,
        Bn_compute_blocks=cache_bn,
        residual_diff_threshold=residual_diff_threshold,
        max_warmup_steps=max_warmup_steps,
        warmup_interval=warmup_interval,
        max_cached_steps=max_cached_steps,
        max_continuous_cached_steps=max_continuous_cached_steps,
        enable_separate_cfg=cache_enable_separate_cfg,
        num_inference_steps=num_inference_steps,
        steps_computation_mask=steps_computation_mask,
        steps_computation_policy=steps_computation_policy,
    )

    cache_config_cls = cache_config.__class__
    params_modifiers = [
        ParamsModifier(
            cache_config=cache_config_cls().reset(
                residual_diff_threshold=residual_diff_threshold,
            ),
        ),
        ParamsModifier(
            cache_config=cache_config_cls().reset(
                residual_diff_threshold=residual_diff_threshold * single_block_rdt_scale,
            ),
        ),
    ]

    calibrator_config = (
        TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        if enable_taylorseer
        else None
    )

    cache_adapter = BlockAdapter(
        pipe=None,
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_3,
        ],
    )

    cache_dit_mod.enable_cache(
        cache_adapter,
        cache_config=cache_config,
        calibrator_config=calibrator_config,
        params_modifiers=params_modifiers,
    )
    pipe._cache_dit_mod = cache_dit_mod
    report = {
        "num_inference_steps": num_inference_steps,
        "steps_mask": "".join(str(v) for v in steps_computation_mask),
        "forced_compute_steps": sum(steps_computation_mask),
        "reuse_candidate_steps": len(steps_computation_mask) - sum(steps_computation_mask),
        "cache_fn": cache_fn,
        "cache_bn": cache_bn,
        "residual_diff_threshold": residual_diff_threshold,
        "single_block_rdt_scale": single_block_rdt_scale,
        "max_warmup_steps": max_warmup_steps,
        "max_cached_steps": max_cached_steps,
        "max_continuous_cached_steps": max_continuous_cached_steps,
        "steps_computation_policy": steps_computation_policy,
        "enable_taylorseer": enable_taylorseer,
        "taylorseer_order": taylorseer_order,
    }
    pipe._cache_dit_policy = report
    return report


# Aliases for apply_attention_backend (same names as flux-stream-editor).
ATTENTION_BACKEND_ALIASES = {
    "fa3": "_flash_3",
    "flash3": "_flash_3",
    "flash_attn_3": "_flash_3",
    "flash-attn-3": "_flash_3",
    "flash_attention_3": "_flash_3",
    "default": "auto",
}
def prepare_transformer_for_speed(
    pipe: Any,
    *,
    backend: str = "sage",
    fuse_qkv: bool = True,
    patch_klein_ops: bool = True,
    direct_nvfp4_dispatch: bool = False,
    fused_qkv_packing: bool = False,
    nvfp4_gemm_backend: str = "torch-scaled-mm",
    static_activation_scales: dict[str, torch.Tensor] | None = None,
    allow_approximate_attention: bool = False,
) -> str | None:
    """
    Apply the stable hot-path configuration once, before cache-dit / compile.

    This keeps the attention processor identity stable while still enabling the
    best available fused QKV path and backend for the current model.
    """
    if patch_klein_ops:
        try:
            from cuda_kernels import is_loaded, load_compiled_extension

            if not is_loaded():
                logger.info("loading klein CUDA extension")
                load_compiled_extension()
            logger.info("applying klein CUDA block patches")
            apply_flux2_transformer_klein_ops(pipe.transformer, verbose=False)
        except Exception as exc:
            logger.warning("klein CUDA block patch unavailable, using diffusers path: %s", exc)
    if (
        fuse_qkv
        and hasattr(pipe.transformer, "fuse_qkv_projections")
        and not getattr(pipe.transformer, "_is_qkv_fused", False)
    ):
        try:
            logger.info("fusing transformer qkv projections")
            pipe.transformer.fuse_qkv_projections()
            setattr(pipe.transformer, "_is_qkv_fused", True)
        except Exception as exc:
            logger.warning("transformer qkv fusion failed, continuing without it: %s", exc)
    elif fuse_qkv and getattr(pipe.transformer, "_is_qkv_fused", False):
        logger.info("transformer qkv projections already fused; preserving existing weights")
    if fuse_qkv:
        try:
            logger.info("fusing double-stream projection stacks")
            fused_blocks = fuse_flux2_double_stream_attention_projections(pipe.transformer)
            if fused_blocks == 0:
                logger.info("no flux2 double-stream projections were fused")
        except Exception as exc:
            logger.warning("double-stream projection fusion failed, continuing without it: %s", exc)
    if direct_nvfp4_dispatch:
        report = patch_torchao_nvfp4_linear_dispatch(
            pipe.transformer,
            gemm_backend=nvfp4_gemm_backend,
            static_activation_scales=static_activation_scales,
        )
        if report["patched"] == 0:
            raise RuntimeError(f"direct TorchAO NVFP4 dispatch requested but no linears were patched: {report}")
    if fused_qkv_packing:
        from cuda_kernels.triton_qkv import install_fused_qkv_packing

        report = install_fused_qkv_packing(pipe.transformer, validate=True)
        if report["patched_double_blocks"] != 5 or report["patched_single_blocks"] != 20:
            raise RuntimeError(f"fused QKV packing did not patch the complete 5+20 block topology: {report}")
    logger.info("selecting attention backend backend=%s", backend)
    return apply_attention_backend(
        pipe,
        backend,
        allow_approximate_attention=allow_approximate_attention,
    )


def _make_fused_linear_from_linears(linears: list[torch.nn.Linear]) -> torch.nn.Linear:
    if not linears:
        raise ValueError("expected at least one linear module")
    first = linears[0]
    if any(linear.in_features != first.in_features for linear in linears):
        raise ValueError("all fused linears must share input features")
    if any(linear.bias is not None for linear in linears) and any(
        linear.bias is None for linear in linears
    ):
        raise ValueError("all fused linears must either all have bias or all omit bias")
    fused = torch.nn.Linear(
        first.in_features,
        sum(linear.out_features for linear in linears),
        bias=first.bias is not None,
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    with torch.no_grad():
        fused.weight.copy_(torch.cat([linear.weight.detach() for linear in linears], dim=0))
        if fused.bias is not None:
            fused.bias.copy_(torch.cat([linear.bias.detach() for linear in linears], dim=0))
    return fused


def calibrate_torchao_nvfp4_activation_scales(
    transformer: Any,
    calibration_fn: Callable[[], None],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Collect static NVFP4 activation scales from representative transformer calls."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    modules = {
        name: module
        for name, module in transformer.named_modules()
        if isinstance(module, torch.nn.Linear) and isinstance(module.weight, NVFP4Tensor)
    }
    if not modules:
        raise RuntimeError("static NVFP4 calibration found no quantized linear modules")

    maxima: dict[str, torch.Tensor] = {}
    call_counts: Counter[str] = Counter()
    handles = []

    def make_hook(name: str):
        def observe(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(f"NVFP4 calibration received invalid input for {name}")
            observed = args[0].detach().abs().amax().to(torch.float32)
            maxima[name] = (
                observed if name not in maxima else torch.maximum(maxima[name], observed)
            )
            call_counts[name] += 1

        return observe

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(make_hook(name)))
        with torch.inference_mode():
            calibration_fn()
    finally:
        for handle in handles:
            handle.remove()

    missing = set(modules).difference(maxima)
    if missing:
        raise RuntimeError(
            "static NVFP4 calibration did not execute every quantized linear: "
            f"{sorted(missing)}"
        )
    scales = {
        name: per_tensor_amax_to_scale(amax).detach()
        for name, amax in maxima.items()
    }
    zero_or_invalid = [
        name
        for name, scale in scales.items()
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all())
    ]
    if zero_or_invalid:
        raise RuntimeError(
            "static NVFP4 calibration produced invalid scales: "
            f"{zero_or_invalid}"
        )
    amax_values = torch.stack(list(maxima.values()))
    report = {
        "module_count": len(modules),
        "total_linear_calls": sum(call_counts.values()),
        "minimum_calls_per_module": min(call_counts.values()),
        "maximum_calls_per_module": max(call_counts.values()),
        "minimum_amax": float(amax_values.min().item()),
        "maximum_amax": float(amax_values.max().item()),
    }
    logger.info("static TorchAO NVFP4 activation calibration complete: %s", report)
    return scales, report


def _direct_torchao_nvfp4_linear_forward(self: torch.nn.Linear, input_tensor: torch.Tensor) -> torch.Tensor:
    """TorchAO-equivalent NVFP4 linear without the Tensor-subclass dispatch layer."""
    if self._klein_nvfp4_gemm_backend == "cutlass-sm120":
        from cuda_kernels.sm120_nvfp4 import linear_forward

        return linear_forward(self, input_tensor)
    if self._klein_nvfp4_dynamic_activation_scale:
        activation_scale = torch.max(torch.abs(input_tensor)).to(torch.float32) / 2688.0
    else:
        activation_scale = self._klein_nvfp4_activation_scale

    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    global_scale = activation_scale.reciprocal() if activation_scale is not None else None
    activation_block_scales, activation_qdata = torch.ops.ao.mslk_quantize_nvfp4(
        input_2d,
        global_scale,
    )

    weight_scale = self._klein_nvfp4_weight_scale
    if activation_scale is not None and weight_scale is not None:
        output_scale = activation_scale * weight_scale
    elif activation_scale is not None:
        output_scale = activation_scale
    else:
        output_scale = weight_scale

    add_bias_separately = (output_scale is not None or input_tensor.dtype == torch.float32) and self.bias is not None
    if self._klein_nvfp4_gemm_backend == "flashinfer-cudnn":
        if output_scale is None:
            raise RuntimeError("FlashInfer cuDNN NVFP4 requires a global output scale")
        from cuda_kernels.flashinfer_nvfp4 import flashinfer_cudnn_fp4_mm

        result = flashinfer_cudnn_fp4_mm(
            activation_qdata,
            self._klein_nvfp4_weight_qdata_t,
            activation_block_scales.view(torch.float8_e4m3fn),
            self._klein_nvfp4_weight_block_scales.view(torch.float8_e4m3fn).t(),
            output_scale.to(torch.float32),
            input_tensor.dtype,
            int(self.out_features),
        )
        output_scale = None  # FlashInfer applies alpha inside the GEMM.
    else:
        result = torch._scaled_mm(
            activation_qdata.view(torch.float4_e2m1fn_x2),
            self._klein_nvfp4_weight_qdata_t.view(torch.float4_e2m1fn_x2),
            activation_block_scales.view(torch.float8_e4m3fn),
            self._klein_nvfp4_weight_block_scales.view(torch.float8_e4m3fn),
            bias=None if add_bias_separately else self.bias,
            out_dtype=input_tensor.dtype,
        )
    if output_scale is not None:
        result = result * output_scale.to(input_tensor.dtype)
    if add_bias_separately:
        result = result + self.bias.to(input_tensor.dtype)
    return result.reshape(*input_tensor.shape[:-1], result.shape[-1])


def patch_torchao_nvfp4_linear_dispatch(
    transformer: Any,
    *,
    strict: bool = True,
    validate: bool = True,
    gemm_backend: str = "torch-scaled-mm",
    static_activation_scales: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """
    Replace TorchAO NVFP4 Tensor-subclass linear dispatch with the equivalent
    raw-tensor MSLK quantize + scaled-mm path before torch.compile().

    Packed weights and scale tables are borrowed directly from NVFP4Tensor.
    No weight conversion, host transfer, or native-owned allocation is
    performed.
    """
    if gemm_backend not in {"torch-scaled-mm", "flashinfer-cudnn", "cutlass-sm120"}:
        raise ValueError(f"unsupported NVFP4 GEMM backend: {gemm_backend}")
    if gemm_backend == "flashinfer-cudnn":
        try:
            import flashinfer
        except Exception as exc:
            raise RuntimeError("FlashInfer cuDNN backend requested but flashinfer is unavailable") from exc
        cuda_version = torch.version.cuda or "0"
        cudnn_version = torch.backends.cudnn.version() or 0
        if int(cuda_version.split(".", maxsplit=1)[0]) < 13 or cudnn_version < 91500:
            raise RuntimeError(
                "FlashInfer cuDNN NVFP4 requires CUDA 13+ and cuDNN 9.15+: "
                f"torch_cuda={cuda_version} cudnn={cudnn_version} flashinfer={flashinfer.__version__}"
            )

    try:
        from torchao.prototype.mx_formats.kernels import mslk_quantize_nvfp4  # noqa: F401
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except Exception as exc:
        raise RuntimeError("TorchAO NVFP4 internals are unavailable") from exc
    kernel_variants: dict[int, str] = {}
    if gemm_backend == "cutlass-sm120":
        if not hasattr(torch.ops, "klein_sm120") or not hasattr(torch.ops.klein_sm120, "nvfp4_gemm_out"):
            raise RuntimeError("SM120 CUTLASS NVFP4 extension is not loaded")
        if not validate:
            raise ValueError("cutlass-sm120 requires setup validation and performance gating")
        from cuda_kernels.sm120_nvfp4 import KERNEL_VARIANTS, prepare_linear, release_linear

        kernel_variants = KERNEL_VARIANTS

    patched = 0
    candidates = 0
    skipped: dict[str, int] = {}
    validated_shapes: set[tuple[int, int, int, bool]] = set()
    validation: list[dict[str, Any]] = []
    shape_backend_choices: dict[tuple[int, int, int, bool], str] = {}
    shape_kernel_variants: dict[tuple[int, int, int, bool], int] = {}
    native_compile_probe: dict[str, Any] | None = None
    native_compile_compatible: bool | None = None
    production_row_counts: Counter[int] = Counter()
    static_scale_names = set(static_activation_scales or {})

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    def production_rows(module_fqn: str) -> int:
        if "single_transformer_blocks" in module_fqn:
            return 2240
        if module_fqn.startswith(
            (
                "time_guidance_embed.",
                "double_stream_modulation_img.",
                "double_stream_modulation_txt.",
                "single_stream_modulation.",
                "norm_out.linear",
            )
        ):
            return 1
        if module_fqn == "context_embedder":
            return 512
        if module_fqn in {"x_embedder", "proj_out"}:
            return 1728
        if module_fqn.startswith("transformer_blocks."):
            if any(
                fragment in module_fqn
                for fragment in (
                    "to_added_qkv",
                    "add_q_proj",
                    "add_k_proj",
                    "add_v_proj",
                    "to_add_out",
                    "ff_context",
                )
            ):
                return 512
            return 1728
        raise RuntimeError(
            "unclassified Klein NVFP4 production shape for native SM120 dispatch: "
            f"module={module_fqn}"
        )

    for module_fqn, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear) or not isinstance(module.weight, NVFP4Tensor):
            continue
        candidates += 1
        weight = module.weight
        quant_kwargs = weight.act_quant_kwargs
        if quant_kwargs is None:
            skip("weight_only")
            continue
        if weight.block_size != 16:
            skip("block_size")
            continue
        if not weight.is_swizzled_scales:
            skip("unblocked_scales")
            continue
        if not quant_kwargs.use_triton_kernel:
            skip("non_mslk_activation_quantizer")
            continue
        if not weight.qdata.is_cuda or not weight.scale.is_cuda:
            skip("non_cuda_weight")
            continue
        if weight.qdata.dtype != torch.uint8:
            skip("qdata_dtype")
            continue
        if weight.scale.dtype != torch.float8_e4m3fn:
            skip("scale_dtype")
            continue
        if not weight.qdata.is_contiguous() or not weight.scale.is_contiguous():
            skip("non_contiguous_weight")
            continue

        buffers = {
            "_klein_nvfp4_weight_qdata_t": weight.qdata.t(),
            "_klein_nvfp4_weight_block_scales": weight.scale,
            "_klein_nvfp4_weight_scale": weight.per_tensor_scale,
            "_klein_nvfp4_activation_scale": (
                static_activation_scales[module_fqn]
                if static_activation_scales is not None
                else weight.act_per_tensor_scale
            ),
        }
        for name, tensor in buffers.items():
            if name in module._buffers:
                module._buffers[name] = tensor
            else:
                module.register_buffer(name, tensor, persistent=False)

        module._klein_nvfp4_dynamic_activation_scale = (
            False
            if static_activation_scales is not None
            else bool(quant_kwargs.use_dynamic_per_tensor_scale)
        )
        validation_rows = 128
        if gemm_backend == "cutlass-sm120":
            validation_rows = production_rows(module_fqn)
            production_row_counts[validation_rows] += 1
        shape_key = (validation_rows, module.in_features, module.out_features, module.bias is not None)
        module._klein_nvfp4_gemm_backend = gemm_backend
        if gemm_backend == "cutlass-sm120":
            selected_for_shape = (
                "torch-scaled-mm"
                if native_compile_compatible is False or module.bias is not None
                else shape_backend_choices.get(shape_key)
            )
            if selected_for_shape == "torch-scaled-mm":
                module._klein_nvfp4_gemm_backend = selected_for_shape
                shape_backend_choices.setdefault(shape_key, selected_for_shape)
            else:
                prepare_linear(module, weight)
                if selected_for_shape == "cutlass-sm120":
                    module._klein_sm120_kernel_variant = shape_kernel_variants[shape_key]
        validate_shape = gemm_backend in {"flashinfer-cudnn", "cutlass-sm120"} or len(validated_shapes) < 3
        if validate and shape_key not in validated_shapes and validate_shape:
            generator = torch.Generator(device=weight.device).manual_seed(0)
            sample = torch.randn(
                (validation_rows, module.in_features),
                generator=generator,
                device=weight.device,
                dtype=weight.orig_dtype,
            )
            with torch.inference_mode():
                dynamic_reference = module(sample)
                if static_activation_scales is not None:
                    requested_backend = module._klein_nvfp4_gemm_backend
                    module._klein_nvfp4_gemm_backend = "torch-scaled-mm"
                    reference = _direct_torchao_nvfp4_linear_forward(module, sample)
                    module._klein_nvfp4_gemm_backend = requested_backend
                else:
                    reference = dynamic_reference
            static_vs_dynamic_max_abs = (
                float((dynamic_reference - reference).abs().max().item())
                if static_activation_scales is not None
                else 0.0
            )
            max_abs = None
            torchao_ms = None
            torchao_samples_ms: list[float] = []
            native_ms = None
            native_speedup = None
            selected_backend = module._klein_nvfp4_gemm_backend
            selected_kernel_variant = None
            kernel_variant_results: list[dict[str, Any]] = []
            if gemm_backend == "cutlass-sm120":
                def benchmark_call(callable_fn: Callable[[], torch.Tensor]) -> tuple[float, list[float]]:
                    for _ in range(5):
                        callable_fn()
                    samples = []
                    for _ in range(5):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):
                            callable_fn()
                        end.record()
                        end.synchronize()
                        samples.append(float(start.elapsed_time(end) / 20.0))
                    return statistics.median(samples), samples

                if static_activation_scales is not None:
                    def torch_scaled_reference_call() -> torch.Tensor:
                        active_backend = module._klein_nvfp4_gemm_backend
                        module._klein_nvfp4_gemm_backend = "torch-scaled-mm"
                        try:
                            return _direct_torchao_nvfp4_linear_forward(module, sample)
                        finally:
                            module._klein_nvfp4_gemm_backend = active_backend

                    torchao_benchmark_call = torch_scaled_reference_call
                else:
                    def dynamic_torchao_reference_call() -> torch.Tensor:
                        return module(sample)

                    torchao_benchmark_call = dynamic_torchao_reference_call
                torchao_ms, torchao_samples_ms = benchmark_call(torchao_benchmark_call)
                valid_variants: list[tuple[float, int, float]] = []
                variants_to_test = (
                    {} if native_compile_compatible is False or module.bias is not None else kernel_variants
                )
                for kernel_variant, tile in variants_to_test.items():
                    module._klein_nvfp4_gemm_backend = "cutlass-sm120"
                    module._klein_sm120_kernel_variant = kernel_variant
                    try:
                        with torch.inference_mode():
                            direct = _direct_torchao_nvfp4_linear_forward(module, sample)
                        variant_max_abs = float((reference - direct).abs().max().item())
                        parity = bool(torch.allclose(reference, direct, rtol=1e-2, atol=1e-2))
                        variant_ms, variant_samples_ms = benchmark_call(
                            lambda: _direct_torchao_nvfp4_linear_forward(module, sample)
                        )
                        if parity:
                            valid_variants.append((variant_ms, kernel_variant, variant_max_abs))
                        kernel_variant_results.append(
                            {
                                "variant": kernel_variant,
                                "tile": tile,
                                "max_abs": variant_max_abs,
                                "parity": parity,
                                "ms": variant_ms,
                                "samples_ms": variant_samples_ms,
                                "error": None,
                            }
                        )
                    except Exception as exc:
                        kernel_variant_results.append(
                            {
                                "variant": kernel_variant,
                                "tile": tile,
                                "max_abs": None,
                                "parity": False,
                                "ms": None,
                                "samples_ms": [],
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                if valid_variants:
                    native_ms, selected_kernel_variant, max_abs = min(valid_variants)
                    native_speedup = torchao_ms / native_ms
                selected_backend = "cutlass-sm120" if (
                    native_ms is not None and native_ms <= torchao_ms * 0.95
                ) else "torch-scaled-mm"
                module._klein_nvfp4_gemm_backend = selected_backend
                shape_backend_choices[shape_key] = selected_backend
                if selected_backend == "cutlass-sm120":
                    module._klein_sm120_kernel_variant = selected_kernel_variant
                    if native_compile_probe is None:
                        try:
                            compiled_probe = torch.compile(
                                lambda tensor: _direct_torchao_nvfp4_linear_forward(module, tensor),
                                mode="reduce-overhead",
                                fullgraph=True,
                                dynamic=False,
                            )
                            with torch.inference_mode():
                                probe_output = compiled_probe(sample).detach().clone()
                                compiled_probe(sample)
                            probe_max_abs = float((reference - probe_output).abs().max().item())
                            probe_parity = bool(
                                torch.allclose(reference, probe_output, rtol=1e-2, atol=1e-2)
                            )
                            if not probe_parity:
                                raise RuntimeError(
                                    f"compiled native parity failed, max_abs={probe_max_abs}"
                                )
                            native_compile_compatible = True
                            native_compile_probe = {
                                "accepted": True,
                                "shape": shape_key,
                                "max_abs": probe_max_abs,
                                "error": None,
                            }
                            logger.info("SM120 compile probe accepted: %s", native_compile_probe)
                        except Exception as exc:
                            native_compile_compatible = False
                            native_compile_probe = {
                                "accepted": False,
                                "shape": shape_key,
                                "max_abs": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            logger.warning(
                                "SM120 compile probe rejected; retaining TorchAO: %s",
                                native_compile_probe,
                            )
                            selected_backend = "torch-scaled-mm"
                            selected_kernel_variant = None
                            module._klein_nvfp4_gemm_backend = selected_backend
                            shape_backend_choices[shape_key] = selected_backend
                            release_linear(module)
                    if selected_backend == "cutlass-sm120":
                        shape_kernel_variants[shape_key] = selected_kernel_variant
                else:
                    release_linear(module)
            else:
                try:
                    direct = _direct_torchao_nvfp4_linear_forward(module, sample)
                except Exception as exc:
                    raise RuntimeError(
                        "NVFP4 backend validation failed "
                        f"backend={gemm_backend} shape={shape_key} "
                        f"error={type(exc).__name__}: {exc}"
                    ) from None
                max_abs = float((reference - direct).abs().max().item())
                if not torch.allclose(reference, direct, rtol=1e-2, atol=1e-2):
                    raise RuntimeError(
                        f"direct NVFP4 dispatch parity failed for shape={shape_key}, max_abs={max_abs}"
                    )
            validation.append(
                {
                    "shape": shape_key,
                    "max_abs": max_abs,
                    "static_vs_dynamic_max_abs": static_vs_dynamic_max_abs,
                    "module": module_fqn,
                    "torchao_ms": torchao_ms,
                    "torchao_samples_ms": torchao_samples_ms,
                    "native_ms": native_ms,
                    "native_speedup": native_speedup,
                    "selected_backend": selected_backend,
                    "selected_kernel_variant": selected_kernel_variant,
                    "kernel_variants": kernel_variant_results,
                }
            )
            logger.info(
                "NVFP4 shape selection shape=%s torchao_ms=%s native_ms=%s "
                "backend=%s tile_variant=%s",
                shape_key,
                torchao_ms,
                native_ms,
                selected_backend,
                selected_kernel_variant,
            )
            validated_shapes.add(shape_key)
        module.forward = MethodType(_direct_torchao_nvfp4_linear_forward, module)
        patched += 1

    if static_activation_scales is not None:
        missing = static_scale_names.difference(name for name, _ in transformer.named_modules())
        if missing:
            raise RuntimeError(
                "static NVFP4 activation scales reference unknown modules: "
                f"{sorted(missing)}"
            )
        if len(static_activation_scales) != patched:
            raise RuntimeError(
                "static NVFP4 activation scale coverage mismatch: "
                f"scales={len(static_activation_scales)} patched={patched}"
            )

    report = {
        "gemm_backend": gemm_backend,
        "candidates": candidates,
        "patched": patched,
        "static_activation_scale_count": (
            len(static_activation_scales) if static_activation_scales is not None else 0
        ),
        "skipped": skipped,
        "validation": validation,
        "shape_backend_choices": {str(key): value for key, value in shape_backend_choices.items()},
        "shape_kernel_variants": {
            str(key): {"variant": value, "tile": kernel_variants[value]}
            for key, value in shape_kernel_variants.items()
        },
        "native_compile_probe": native_compile_probe,
        "production_row_counts": dict(production_row_counts),
        "selected_backend_counts": dict(
            Counter(
                getattr(module, "_klein_nvfp4_gemm_backend", "unpatched")
                for module in transformer.modules()
                if isinstance(module, torch.nn.Linear) and isinstance(module.weight, NVFP4Tensor)
            )
        ),
    }
    transformer._klein_nvfp4_dispatch_report = report
    logger.info("direct TorchAO NVFP4 dispatch patch: %s", report)
    if strict and patched != candidates:
        raise RuntimeError(f"direct NVFP4 dispatch did not patch every candidate: {report}")
    return report


def fuse_flux2_double_stream_attention_projections(transformer: Any) -> int:
    """
    Replace the double-stream Flux2Attention projection stacks with fused qkv / added-qkv
    modules when the block shape matches Diffusers Flux2Attention.

    This keeps the model semantics unchanged but collapses 6 GEMMs into 2 GEMMs per
    double-stream attention block, which is the biggest remaining linear overhead in the
    current checkpoint.
    """
    fused_blocks = 0
    for module in transformer.modules():
        if module.__class__.__name__ != "Flux2Attention":
            continue
        if not all(hasattr(module, name) for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")):
            continue
        if getattr(module, "fused_projections", False):
            continue
        try:
            module.to_qkv = _make_fused_linear_from_linears([module.to_q, module.to_k, module.to_v])
            module.to_added_qkv = _make_fused_linear_from_linears(
                [module.add_q_proj, module.add_k_proj, module.add_v_proj]
            )
            module.fused_projections = True
            # Keep the original modules attached for state_dict compatibility but stop
            # the hot path from touching them through the fused projection branch.
            fused_blocks += 1
        except Exception:
            continue
    if fused_blocks:
        print(f"flux2_double_stream_attention_fused_blocks={fused_blocks}")
    return fused_blocks


def _benchmark_attention_backends(
    transformer: Any,
    *,
    allow_approximate_attention: bool,
) -> tuple[str, dict[str, Any]]:
    import flashinfer

    from cuda_kernels.flashinfer_attention import (
        FLASHINFER_ATTENTION_BACKEND,
        FLASHINFER_FP16_REDUCTION_ATTENTION_BACKEND,
        FLASHINFER_NVFP4_ATTENTION_BACKEND,
        flashinfer_exact_attention,
        flashinfer_fp16_reduction_attention,
        flashinfer_nvfp4_attention,
    )
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    sequence = 2240
    heads = 24
    head_dim = 128
    device = next(transformer.parameters()).device
    generator = torch.Generator(device=device).manual_seed(0)
    query = torch.randn(
        (1, sequence, heads, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        query.shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        query.shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    def benchmark_call(callable_fn: Callable[[], torch.Tensor]) -> tuple[float, list[float]]:
        for _ in range(5):
            callable_fn()
        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                callable_fn()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end) / 20.0))
        return statistics.median(samples), samples

    with torch.inference_mode():
        reference = dispatch_attention_fn(query, key, value, backend="native").detach().clone()
    results: list[dict[str, Any]] = []
    valid: list[tuple[float, str]] = []
    flashinfer_processor_types = {
        "KleinFusedQKVPackingProcessor",
        "KleinFusedSingleQKVPackingProcessor",
    }
    flashinfer_processor_count = sum(
        1
        for module in transformer.modules()
        if getattr(module, "processor", None).__class__.__name__
        in flashinfer_processor_types
    )
    candidate_calls: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("native", lambda: dispatch_attention_fn(query, key, value, backend="native")),
        ("_flash_3", lambda: dispatch_attention_fn(query, key, value, backend="_flash_3")),
    ]
    if flashinfer_processor_count == 25:
        candidate_calls.append(
            (
                FLASHINFER_ATTENTION_BACKEND,
                lambda: flashinfer_exact_attention(query, key, value),
            )
        )
    if allow_approximate_attention and flashinfer_processor_count == 25:
        candidate_calls.extend(
            [
                (
                    FLASHINFER_FP16_REDUCTION_ATTENTION_BACKEND,
                    lambda: flashinfer_fp16_reduction_attention(query, key, value),
                ),
                (
                    FLASHINFER_NVFP4_ATTENTION_BACKEND,
                    lambda: flashinfer_nvfp4_attention(query, key, value),
                ),
            ]
        )

    approximate_limits = {
        FLASHINFER_FP16_REDUCTION_ATTENTION_BACKEND: (0.05, 0.995),
        FLASHINFER_NVFP4_ATTENTION_BACKEND: (0.15, 0.98),
    }
    for candidate, callable_fn in candidate_calls:
        try:
            with torch.inference_mode():
                output = callable_fn().detach().clone()
            output_f32 = output.float()
            reference_f32 = reference.float()
            relative_l2 = float(
                torch.linalg.vector_norm(output_f32 - reference_f32)
                / torch.linalg.vector_norm(reference_f32).clamp_min(1e-12)
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    output_f32.flatten(), reference_f32.flatten(), dim=0
                )
            )
            is_approximate = candidate in approximate_limits
            relative_l2_limit, cosine_limit = approximate_limits.get(candidate, (0.01, 0.999))
            parity = bool(
                torch.isfinite(output_f32).all()
                and relative_l2 <= relative_l2_limit
                and cosine >= cosine_limit
            )
            median_ms, samples_ms = benchmark_call(callable_fn)
            if parity:
                valid.append((median_ms, candidate))
            results.append(
                {
                    "backend": candidate,
                    "median_ms": median_ms,
                    "samples_ms": samples_ms,
                    "relative_l2": relative_l2,
                    "cosine": cosine,
                    "parity": parity,
                    "approximate": is_approximate,
                    "relative_l2_limit": relative_l2_limit,
                    "cosine_limit": cosine_limit,
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "backend": candidate,
                    "median_ms": None,
                    "samples_ms": [],
                    "relative_l2": None,
                    "cosine": None,
                    "parity": False,
                    "approximate": candidate in approximate_limits,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    native_result = next(item for item in results if item["backend"] == "native")
    if native_result["median_ms"] is None:
        raise RuntimeError(f"native attention benchmark failed: {native_result}")
    fastest_ms, fastest_backend = min(valid)
    selected = (
        fastest_backend
        if fastest_backend != "native" and fastest_ms <= native_result["median_ms"] * 0.95
        else "native"
    )
    compile_probe = None
    flashinfer_backends = {
        FLASHINFER_ATTENTION_BACKEND: flashinfer_exact_attention,
        FLASHINFER_FP16_REDUCTION_ATTENTION_BACKEND: flashinfer_fp16_reduction_attention,
        FLASHINFER_NVFP4_ATTENTION_BACKEND: flashinfer_nvfp4_attention,
    }
    if selected in flashinfer_backends:
        try:
            selected_attention_fn = flashinfer_backends[selected]
            compiled = torch.compile(
                selected_attention_fn,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            )
            with torch.inference_mode():
                compiled_output = compiled(query, key, value).detach().clone()
                compiled(query, key, value)
            compiled_f32 = compiled_output.float()
            relative_l2 = float(
                torch.linalg.vector_norm(compiled_f32 - reference.float())
                / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    compiled_f32.flatten(), reference.float().flatten(), dim=0
                )
            )
            relative_l2_limit, cosine_limit = approximate_limits.get(selected, (0.01, 0.999))
            if relative_l2 > relative_l2_limit or cosine < cosine_limit:
                raise RuntimeError(
                    f"compiled attention parity failed relative_l2={relative_l2} cosine={cosine}"
                )
            compile_probe = {
                "accepted": True,
                "relative_l2": relative_l2,
                "cosine": cosine,
                "error": None,
            }
        except Exception as exc:
            selected = "native"
            compile_probe = {
                "accepted": False,
                "relative_l2": None,
                "cosine": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    report = {
        "shape": [1, sequence, heads, head_dim],
        "results": results,
        "selected": selected,
        "minimum_required_speedup": 1.05,
        "allow_approximate_attention": allow_approximate_attention,
        "flashinfer_processor_count": flashinfer_processor_count,
        "flashinfer_version": str(getattr(flashinfer, "__version__", "unknown")),
        "compile_probe": compile_probe,
    }
    transformer._klein_attention_selection_report = report
    logger.info("attention backend selection: %s", report)
    return selected, report


def _set_flashinfer_attention_backend(transformer: Any, backend: str) -> int:
    patched = 0
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        if processor is not None and processor.__class__.__name__ in {
            "KleinFusedQKVPackingProcessor",
            "KleinFusedSingleQKVPackingProcessor",
        }:
            processor._attention_backend = backend
            patched += 1
    if patched != 25:
        raise RuntimeError(f"expected 25 Klein attention processors, patched {patched}")
    return patched


def apply_attention_backend(
    pipe: Any,
    backend: str = "sage",
    *,
    allow_approximate_attention: bool = False,
) -> str | None:
    """
    Set the transformer attention backend. Nothing is automatic: you must call this
    after loading the pipeline; installing flash-attn or sage alone is not enough.

    backend: "sage" | "native" | "_flash_3" | "fa3" (alias for _flash_3) | "auto".
    - "auto": benchmark backends at Klein's production attention shape and
      select a numerically valid candidate only when it is at least 5% faster.
      Approximate candidates participate only when explicitly allowed.
    - Otherwise set the given backend (after resolving aliases).

    Returns the backend name that was set, or None if the transformer does not
    support set_attention_backend or all candidates failed (auto).
    """
    if not hasattr(pipe.transformer, "set_attention_backend"):
        return None
    resolved = (backend or "").strip().lower()
    if resolved in {"cudnn", "cudnn-sdpa", "cudnn_attention"}:
        try:
            torch.backends.cuda.enable_cudnn_sdp(True)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            pipe.transformer.set_attention_backend("native")
            logger.info("forced exact cuDNN SDPA attention backend")
            return "cudnn-sdpa"
        except Exception as exc:
            logger.warning("cuDNN SDPA attention backend unavailable: %s", exc)
            return None
    resolved = ATTENTION_BACKEND_ALIASES.get(resolved, resolved)
    if resolved == "auto":
        selected, report = _benchmark_attention_backends(
            pipe.transformer,
            allow_approximate_attention=allow_approximate_attention,
        )
        if selected.startswith("flashinfer-"):
            patched = _set_flashinfer_attention_backend(pipe.transformer, selected)
            report["patched_processors"] = patched
        else:
            pipe.transformer.set_attention_backend(selected)
        print(f"attention_backend_selection={json.dumps(report, sort_keys=True)}")
        return selected
    try:
        pipe.transformer.set_attention_backend(resolved)
        return resolved
    except Exception:
        return None


def apply_transformer_compile(
    pipe: Any,
    *,
    disable_cudagraphs: bool = True,
    mode: str = "reduce-overhead",
) -> None:
    """
    Compile the pipeline transformer with torch.compile. Works with or without cache-dit.
    If cache-dit is enabled (pipe._cache_dit_mod), calls set_compile_configs() first so
    cache and compile interoperate correctly. First run after this will be slow (compilation).

    Defaults mirror flux-stream-editor: disable_cudagraphs=True (options triton.cudagraphs=False),
    mode="reduce-overhead" when cudagraphs are enabled. PyTorch recommends "reduce-overhead" for
    inference (reduces Python overhead; "max-autotune" compiles longer for marginal gain).
    """
    if getattr(pipe, "_cache_dit_mod", None) is not None:
        pipe._cache_dit_mod.set_compile_configs()
    kwargs = {"fullgraph": False}
    if disable_cudagraphs:
        kwargs["options"] = {"triton.cudagraphs": False}
    else:
        kwargs["mode"] = mode
    pipe.transformer = torch.compile(pipe.transformer, **kwargs)


def enable_cuda_denoiser_op(
    pipe: Any,
    op_path: str = "klein_cuda.denoise_step",
    *,
    expected_hidden_tokens: int | None = None,
    enforce_cfg1: bool = False,
) -> None:
    """
    Register a torch.ops-backed CUDA denoiser callback for pipeline denoising.

    Expected op signature:
      denoise_step(hidden_states, timestep, encoder_hidden_states, txt_ids, img_ids) -> noise_pred

    Args:
      expected_hidden_tokens: Optional sequence-length guard for fixed-shape inference.
      enforce_cfg1: If True, reject the "uncond" path (requires guidance_scale == 1.0).
    """
    if not hasattr(pipe, "set_cuda_denoiser"):
        raise AttributeError("Pipeline does not support custom CUDA denoiser registration")
    namespace, op_name = op_path.rsplit(".", 1)
    op_ns = getattr(torch.ops, namespace, None)
    if op_ns is None:
        raise RuntimeError(f"torch.ops namespace not found: {namespace}")
    op = getattr(op_ns, op_name, None)
    if op is None:
        raise RuntimeError(f"torch.ops op not found: {op_path}")

    def _denoiser(
        *,
        transformer: Any,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None,
        context: str,
    ) -> torch.Tensor:
        del transformer, joint_attention_kwargs
        if enforce_cfg1 and context == "uncond":
            raise RuntimeError("CUDA denoiser configured with enforce_cfg1=True but uncond path was requested")
        if expected_hidden_tokens is not None and hidden_states.shape[1] != expected_hidden_tokens:
            raise RuntimeError(
                f"CUDA denoiser expected hidden token length {expected_hidden_tokens}, got {hidden_states.shape[1]}",
            )
        out = op(hidden_states, timestep / 1000, encoder_hidden_states, txt_ids, img_ids)
        if out.shape != hidden_states.shape:
            raise RuntimeError(
                f"CUDA denoiser output shape mismatch: expected {tuple(hidden_states.shape)}, got {tuple(out.shape)}",
            )
        return out

    pipe.set_cuda_denoiser(_denoiser, name=op_path)
