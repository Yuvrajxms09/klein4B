"""
Cache-DiT (DBCache) integration for FLUX.2 Klein 4B.

Defaults match flux-stream-editor (FastFlux2Config / build_default_config):
cache_fn=1, cache_bn=0, residual_diff_threshold=0.8, single_block_rdt_scale=3.0,
max_warmup_steps=0, steps_mask by step count ("10" for 2 steps, "1"*n for n steps).
Call enable_cache_dit(pipe) after loading; __call__ will refresh_context before each run.
"""

from __future__ import annotations

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
    loading (e.g. after from_pretrained, optionally after replace_pipeline_vae_with_taef2).

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
