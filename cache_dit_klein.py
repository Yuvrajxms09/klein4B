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
