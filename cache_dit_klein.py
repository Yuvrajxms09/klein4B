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

import torch
from typing import Any
from types import MethodType

try:
    from einops import rearrange
except Exception:  # pragma: no cover - optional dependency in some envs
    rearrange = None


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
        raise RuntimeError("torch.ops.klein_cuda is unavailable; load the extension first")

    def _split_qkv_heads(qkv: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, _ = qkv.shape
        head_dim = qkv.shape[-1] // (3 * num_heads)
        qkv = qkv.view(b, l, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        return qkv[0], qkv[1], qkv[2]

    def _split_modulation(mod: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        parts = mod.chunk(6, dim=-1)
        first = parts[:3]
        second = parts[3:]
        if first[0].ndim == 2:
            first = tuple(t[:, None, :] for t in first)
            second = tuple(t[:, None, :] for t in second)
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

    def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        q, k = _apply_rope(q, k, pe)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(x, "B H L D -> B L (H D)")

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
            qw = module.query_norm.scale.float().contiguous()
            kw = module.key_norm.scale.float().contiguous()
            qn = ns.qk_rms_norm_(qn, kn, qw, kw)
            q = qn.reshape(q.shape[0], q.shape[2], q.shape[1], q.shape[3]).permute(0, 2, 1, 3).to(v)
            k = kn.reshape(k.shape[0], k.shape[2], k.shape[1], k.shape[3]).permute(0, 2, 1, 3).to(v)
            return q, k
        return module.query_norm(q).to(v), module.key_norm(k).to(v)

    def _silu_mul(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        if x1.is_cuda and x1.dtype == torch.float32 and x1.shape == x2.shape:
            return ns.silu_mul_(x1, x2)
        return torch.nn.functional.silu(x1) * x2

    def _single_forward(self, x: torch.Tensor, pe: torch.Tensor, mod: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = mod
        x_mod = self.pre_norm(x)
        x_mod = x_mod.mul_(1 + mod_scale).add_(mod_shift)
        fused = self.linear1(x_mod)
        qkv = fused[..., : 3 * self.hidden_size]
        mlp = fused[..., 3 * self.hidden_size :]
        q, k, v = _split_qkv_heads(qkv, self.num_heads)
        q, k = _qk_norm(self.norm, q, k, v)
        attn = _attention(q, k, v, pe)
        output = self.linear2(torch.cat((attn, _silu_mul(mlp)), dim=-1))
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
        if verbose:
            print("[klein] single block", type(self).__name__, "hidden", tuple(hidden_states.shape))
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        mod_shift, mod_scale, mod_gate = temb_mod.chunk(3, dim=-1)
        if mod_shift.ndim == 2:
            mod_shift = mod_shift[:, None, :]
            mod_scale = mod_scale[:, None, :]
            mod_gate = mod_gate[:, None, :]
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        attn_output = self.attn(
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
        img_q, img_k, img_v = _split_qkv_heads(self.img_attn.qkv(img_modulated), self.num_heads)
        img_q, img_k = _qk_norm(self.img_attn.norm, img_q, img_k, img_v)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = txt_modulated.mul_(1 + txt_mod1_scale).add_(txt_mod1_shift)
        txt_q, txt_k, txt_v = _split_qkv_heads(self.txt_attn.qkv(txt_modulated), self.num_heads)
        txt_q, txt_k = _qk_norm(self.txt_attn.norm, txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        pe = torch.cat((pe_ctx, pe), dim=2)
        attn = _attention(q, k, v, pe)
        txt_attn, img_attn = attn[:, : txt_q.shape[2]], attn[:, txt_q.shape[2] :]

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

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        # Use the existing attention module, but keep the block math and projections in-place.
        attn_hidden, attn_context = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * attn_context
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
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
) -> None:
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


# Aliases for apply_attention_backend (same names as flux-stream-editor).
ATTENTION_BACKEND_ALIASES = {
    "fa3": "_flash_3",
    "flash3": "_flash_3",
    "flash_attn_3": "_flash_3",
    "flash-attn-3": "_flash_3",
    "flash_attention_3": "_flash_3",
    "default": "auto",
}
# Order tried when backend="auto": prefer flash3, then sage, then native.
AUTO_ATTENTION_BACKEND_CANDIDATES = ("_flash_3", "sage", "native")


def apply_attention_backend(pipe: Any, backend: str = "sage") -> str | None:
    """
    Set the transformer attention backend. Nothing is automatic: you must call this
    after loading the pipeline; installing flash-attn or sage alone is not enough.

    backend: "sage" | "native" | "_flash_3" | "fa3" (alias for _flash_3) | "auto".
    - "auto": try _flash_3, then sage, then native; use first that succeeds.
    - Otherwise set the given backend (after resolving aliases).

    Returns the backend name that was set, or None if the transformer does not
    support set_attention_backend or all candidates failed (auto).
    """
    if not hasattr(pipe.transformer, "set_attention_backend"):
        return None
    resolved = (backend or "").strip().lower()
    resolved = ATTENTION_BACKEND_ALIASES.get(resolved, resolved)
    if resolved == "auto":
        for candidate in AUTO_ATTENTION_BACKEND_CANDIDATES:
            try:
                pipe.transformer.set_attention_backend(candidate)
                return candidate
            except Exception:
                continue
        return None
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
