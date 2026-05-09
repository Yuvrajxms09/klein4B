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

import logging
import torch
import torch.nn.functional as F
from typing import Any
from types import MethodType
from pathlib import Path
from typing import Optional

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention as DiffusersFlux2Attention,
    Flux2AttnProcessor as DiffusersFlux2AttnProcessor,
    Flux2KVAttnProcessor as DiffusersFlux2KVAttnProcessor,
    Flux2KVParallelSelfAttnProcessor as DiffusersFlux2KVParallelSelfAttnProcessor,
    Flux2ParallelSelfAttention as DiffusersFlux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor as DiffusersFlux2ParallelSelfAttnProcessor,
    _flux2_kv_causal_attention,
    _get_qkv_projections,
)

try:  # diffusers private helper changed across releases
    from diffusers.models.transformers.transformer_flux2 import _cat_into_buffer
except Exception:  # pragma: no cover - compatibility fallback
    def _cat_into_buffer(module: Any, name: str, parts: tuple[torch.Tensor, ...], dim: int = -1) -> torch.Tensor:
        del module, name
        return torch.cat(parts, dim=dim)

try:
    from einops import rearrange
except Exception:  # pragma: no cover - optional dependency in some envs
    rearrange = None


logger = logging.getLogger(__name__)


def _get_transformer_config(pipe: Any) -> Any:
    config = getattr(pipe, "_transformer_config", None)
    if config is not None:
        return config
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        config = getattr(transformer, "config", None)
        if config is not None:
            return config
    raise AttributeError("pipe transformer config is unavailable")


def latent_token_count_for_resolution(pipe: Any, height: int, width: int) -> int:
    """
    Compute packed latent token length (H*W in packed latent space) used by the denoiser.
    """
    multiple_of = int(pipe.vae_scale_factor) * 2
    h = 2 * (int(height) // multiple_of)
    w = 2 * (int(width) // multiple_of)
    return (h // 2) * (w // 2)


class SpatialCache:
    """
    Per-layer spatial KV cache and output cache for temporal consistency.

    The cache is keyed by the attention module identity, so the wrapper can be
    reused without depending on block indices.
    """

    def __init__(
        self,
        image_seq_len: int,
        text_seq_len: int = 512,
        output_channels: int = 128,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.image_seq_len = int(image_seq_len)
        self.text_seq_len = int(text_seq_len)
        self.full_seq_len = self.text_seq_len + self.image_seq_len
        self.output_channels = int(output_channels)
        self.attention_head_dim = int(attention_head_dim)
        self.num_attention_heads = int(num_attention_heads)
        self.device = device
        self.dtype = dtype
        self.step_key: Any = "default"
        self._step_states: dict[Any, dict[str, Any]] = {}
        logger.info(
            "spatial cache init image_seq_len=%s text_seq_len=%s output_channels=%s heads=%s head_dim=%s device=%s dtype=%s",
            self.image_seq_len,
            self.text_seq_len,
            self.output_channels,
            self.num_attention_heads,
            self.attention_head_dim,
            self.device,
            self.dtype,
        )

    def set_step_key(self, step_key: Any) -> None:
        if step_key != self.step_key and logger.isEnabledFor(logging.DEBUG):
            logger.debug("spatial cache step key change old=%s new=%s", self.step_key, step_key)
        self.step_key = step_key

    def _get_state(self) -> dict[str, Any]:
        state = self._step_states.get(self.step_key)
        if state is None:
            state = {
                "output_cache": torch.zeros(
                    1, self.image_seq_len, self.output_channels, device=self.device, dtype=self.dtype
                ),
                "valid": torch.zeros(1, self.full_seq_len, device=self.device, dtype=torch.bool),
                "double_block_keys": {},
                "double_block_values": {},
                "single_block_keys": {},
                "single_block_values": {},
            }
            self._step_states[self.step_key] = state
        return state

    @staticmethod
    def _mask_counts(mask: torch.Tensor) -> dict[str, int]:
        return {
            "skip": int((mask == 0).sum().item()),
            "execute": int((mask == 1).sum().item()),
            "update": int((mask == 2).sum().item()),
        }

    def clear(self, step_key: Any | None = None) -> None:
        if step_key is None:
            self._step_states.clear()
            logger.info("spatial cache cleared all step states")
            return
        self._step_states.pop(step_key, None)
        logger.info("spatial cache cleared step state step_key=%s", step_key)

    def preprocess_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        state = self._get_state()
        if input_mask.ndim != 2:
            raise ValueError(f"input_mask must be rank 2, got shape {tuple(input_mask.shape)}")

        if input_mask.shape[1] == self.full_seq_len:
            normalized_mask = input_mask
        elif input_mask.shape[1] == self.text_seq_len + 2 * self.image_seq_len:
            text_mask = input_mask[:, : self.text_seq_len]
            image_mask_a = input_mask[:, self.text_seq_len : self.text_seq_len + self.image_seq_len]
            image_mask_b = input_mask[
                :, self.text_seq_len + self.image_seq_len : self.text_seq_len + 2 * self.image_seq_len
            ]
            normalized_mask = torch.cat(
                [text_mask, torch.maximum(image_mask_a, image_mask_b)],
                dim=-1,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "spatial cache normalized mask layout step_key=%s input_len=%s output_len=%s",
                    self.step_key,
                    input_mask.shape[1],
                    normalized_mask.shape[1],
                )
        else:
            raise ValueError(
                "unexpected mask length: got "
                f"{input_mask.shape[1]}, expected {self.full_seq_len} or "
                f"{self.text_seq_len + 2 * self.image_seq_len}"
            )

        processed = torch.where(
            state["valid"] == 0,
            torch.tensor(2, device=self.device, dtype=torch.int32),
            normalized_mask,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "spatial cache preprocess mask step_key=%s counts_in=%s counts_out=%s valid=%s",
                self.step_key,
                self._mask_counts(normalized_mask),
                self._mask_counts(processed),
                int(state["valid"].sum().item()),
            )
        return processed

    def sync_with_output_cache(self, mask: torch.Tensor, masked_prediction: torch.Tensor) -> torch.Tensor:
        state = self._get_state()
        image_mask = mask[:, self.text_seq_len :]
        execute_mask = image_mask != 0
        update_mask = image_mask == 2

        execute_exp = execute_mask.unsqueeze(-1)
        update_exp = update_mask.unsqueeze(-1)

        output_cache = state["output_cache"]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "spatial cache output sync step_key=%s mask_counts=%s prediction_shape=%s cache_shape=%s",
                self.step_key,
                self._mask_counts(mask),
                tuple(masked_prediction.shape),
                tuple(output_cache.shape),
            )
        filled_prediction = torch.where(execute_exp, masked_prediction, output_cache)
        state["output_cache"] = torch.where(update_exp, masked_prediction, output_cache)
        state["valid"] = torch.logical_or(state["valid"], mask == 2)
        return filled_prediction

    def sync_with_kv_cache(
        self,
        mask: torch.Tensor,
        masked_keys: torch.Tensor,
        masked_values: torch.Tensor,
        block_key: Any,
        *,
        block_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._get_state()
        execute_mask = mask != 0
        update_mask = mask == 2
        execute_exp = execute_mask.unsqueeze(-1).unsqueeze(-1)
        update_exp = update_mask.unsqueeze(-1).unsqueeze(-1)

        key_id = id(block_key)
        if block_type == "single":
            cached_keys = state["single_block_keys"].get(key_id)
            cached_values = state["single_block_values"].get(key_id)
        elif block_type == "double":
            cached_keys = state["double_block_keys"].get(key_id)
            cached_values = state["double_block_values"].get(key_id)
        else:
            raise ValueError(f"block_type must be 'single' or 'double', got {block_type}")

        if cached_keys is None or cached_keys.shape != masked_keys.shape:
            cached_keys = torch.zeros_like(masked_keys)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("spatial cache kv miss step_key=%s block_type=%s key_id=%s", self.step_key, block_type, key_id)
        if cached_values is None or cached_values.shape != masked_values.shape:
            cached_values = torch.zeros_like(masked_values)

        filled_keys = torch.where(execute_exp, masked_keys, cached_keys)
        filled_values = torch.where(execute_exp, masked_values, cached_values)
        updated_keys = torch.where(update_exp, masked_keys, cached_keys)
        updated_values = torch.where(update_exp, masked_values, cached_values)

        if block_type == "single":
            state["single_block_keys"][key_id] = updated_keys
            state["single_block_values"][key_id] = updated_values
        else:
            state["double_block_keys"][key_id] = updated_keys
            state["double_block_values"][key_id] = updated_values

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "spatial cache kv sync step_key=%s block_type=%s key_id=%s mask_counts=%s key_shape=%s value_shape=%s",
                self.step_key,
                block_type,
                key_id,
                self._mask_counts(mask),
                tuple(masked_keys.shape),
                tuple(masked_values.shape),
            )
        return filled_keys, filled_values


class Flux2KVLayerCache:
    """Per-layer KV cache for reference image tokens."""

    def __init__(self):
        self.k_ref: torch.Tensor | None = None
        self.v_ref: torch.Tensor | None = None

    def store(self, k_ref: torch.Tensor, v_ref: torch.Tensor) -> None:
        self.k_ref = k_ref
        self.v_ref = v_ref

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_ref is None or self.v_ref is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k_ref, self.v_ref

    def clear(self) -> None:
        self.k_ref = None
        self.v_ref = None


class Flux2KVCache:
    """Container for all layers' reference-token KV caches."""

    def __init__(self, num_double_layers: int, num_single_layers: int):
        self.double_block_caches = [Flux2KVLayerCache() for _ in range(num_double_layers)]
        self.single_block_caches = [Flux2KVLayerCache() for _ in range(num_single_layers)]
        self.num_ref_tokens: int = 0

    def get_double(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.double_block_caches[layer_idx]

    def get_single(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.single_block_caches[layer_idx]

    def clear(self) -> None:
        for cache in self.double_block_caches:
            cache.clear()
        for cache in self.single_block_caches:
            cache.clear()
        self.num_ref_tokens = 0


def sparse_mlp_compute(
    mlp_function: Any,
    mask: torch.Tensor | None,
    input_hidden_states: torch.Tensor,
    output_dims: int | None = None,
):
    if mask is None:
        return mlp_function(input_hidden_states)

    seq_mask = mask.squeeze(0)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "sparse mlp compute module=%s input_shape=%s mask_counts=%s",
            getattr(mlp_function, "__class__", type(mlp_function)).__name__,
            tuple(input_hidden_states.shape),
            SpatialCache._mask_counts(mask),
        )
    if seq_mask.any():
        active_idx = seq_mask.nonzero(as_tuple=False).squeeze(-1)
        mlp_active = input_hidden_states.index_select(1, active_idx)
        mlp_active_out = mlp_function(mlp_active)
        mlp_out = torch.zeros(
            1,
            input_hidden_states.shape[1],
            mlp_active_out.shape[-1],
            device=input_hidden_states.device,
            dtype=input_hidden_states.dtype,
        )
        mlp_out.index_copy_(1, active_idx, mlp_active_out)
        return mlp_out

    if output_dims is None:
        sample_out = mlp_function(input_hidden_states[:, :1])
        output_dims = sample_out.shape[-1]

    return torch.zeros(
        1,
        input_hidden_states.shape[1],
        output_dims,
        device=input_hidden_states.device,
        dtype=input_hidden_states.dtype,
    )


def sparse_attention_compute(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask,
    backend,
    parallel_config,
    query_mask: torch.Tensor | None = None,
):
    if query_mask is None:
        return dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            backend=backend,
            parallel_config=parallel_config,
        )

    hidden_states = torch.zeros_like(query)
    seq_mask = query_mask.squeeze(0)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "sparse attention compute query_shape=%s key_shape=%s value_shape=%s mask_counts=%s backend=%s",
            tuple(query.shape),
            tuple(key.shape),
            tuple(value.shape),
            SpatialCache._mask_counts(query_mask),
            backend,
        )
    if seq_mask.any():
        active_idx = seq_mask.nonzero(as_tuple=False).squeeze(-1)
        query_active = query.index_select(1, active_idx)
        hidden_states_active = dispatch_attention_fn(
            query_active,
            key,
            value,
            attn_mask=attn_mask,
            backend=backend,
            parallel_config=parallel_config,
        )
        hidden_states.index_copy_(1, active_idx, hidden_states_active)
    return hidden_states


def _apply_spatial_kv_cache(
    spatial_cache: SpatialCache | None,
    mask: torch.Tensor | None,
    key: torch.Tensor,
    value: torch.Tensor,
    attn: Any,
    block_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spatial_cache is None or mask is None:
        return key, value
    return spatial_cache.sync_with_kv_cache(mask, key, value, attn, block_type=block_type)


def _temporal_qkv_projections(
    attn: DiffusersFlux2Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if mask is None or getattr(attn, "fused_projections", False):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "temporal qkv projections using dense path module=%s fused=%s mask_present=%s",
                attn.__class__.__name__,
                bool(getattr(attn, "fused_projections", False)),
                mask is not None,
            )
        return _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

    text_seq_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "temporal qkv projections sparse path module=%s hidden_shape=%s text_seq_len=%s mask_counts=%s",
            attn.__class__.__name__,
            tuple(hidden_states.shape),
            text_seq_len,
            SpatialCache._mask_counts(mask),
        )
    query = sparse_mlp_compute(attn.to_q, mask[:, text_seq_len:], hidden_states, attn.to_q.out_features)
    key = sparse_mlp_compute(attn.to_k, mask[:, text_seq_len:], hidden_states, attn.to_k.out_features)
    value = sparse_mlp_compute(attn.to_v, mask[:, text_seq_len:], hidden_states, attn.to_v.out_features)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = sparse_mlp_compute(
            attn.add_q_proj,
            mask[:, :text_seq_len],
            encoder_hidden_states,
            attn.add_q_proj.out_features,
        )
        encoder_key = sparse_mlp_compute(
            attn.add_k_proj,
            mask[:, :text_seq_len],
            encoder_hidden_states,
            attn.add_k_proj.out_features,
        )
        encoder_value = sparse_mlp_compute(
            attn.add_v_proj,
            mask[:, :text_seq_len],
            encoder_hidden_states,
            attn.add_v_proj.out_features,
        )

    return query, key, value, encoder_query, encoder_key, encoder_value


class TemporalFlux2AttnProcessor(DiffusersFlux2AttnProcessor):
    def __call__(
        self,
        attn: DiffusersFlux2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
        spatial_cache: SpatialCache | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _temporal_qkv_projections(
            attn, hidden_states, encoder_hidden_states, mask
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "temporal attn processor module=%s kv_mode=%s num_ref_tokens=%s mask_present=%s",
                attn.__class__.__name__,
                kv_cache_mode,
                num_ref_tokens,
                mask is not None,
            )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = _cat_into_buffer(attn, "_joint_query_buffer", (encoder_query, query), dim=1)
            key = _cat_into_buffer(attn, "_joint_key_buffer", (encoder_key, key), dim=1)
            value = _cat_into_buffer(attn, "_joint_value_buffer", (encoder_value, value), dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        key, value = _apply_spatial_kv_cache(spatial_cache, mask, key, value, attn, "double")

        num_txt_tokens = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "temporal attn kv extract stored module=%s ref_range=[%s,%s) key_shape=%s value_shape=%s",
                    attn.__class__.__name__,
                    ref_start,
                    ref_end,
                    tuple(key[:, ref_start:ref_end].shape),
                    tuple(value[:, ref_start:ref_end].shape),
                )

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "temporal attn kv cached lookup module=%s num_txt_tokens=%s num_ref_tokens=%s",
                    attn.__class__.__name__,
                    num_txt_tokens,
                    getattr(kv_cache, "num_ref_tokens", None),
                )
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            hidden_states = sparse_attention_compute(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
                query_mask=mask,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "temporal attn dense/sparse attention completed module=%s out_shape=%s",
                    attn.__class__.__name__,
                    tuple(hidden_states.shape),
                )

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            if mask is not None:
                encoder_hidden_states = sparse_mlp_compute(
                    attn.to_add_out,
                    mask[:, : encoder_hidden_states.shape[1]],
                    encoder_hidden_states,
                    attn.to_add_out.out_features,
                )
            else:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if mask is not None:
            image_mask = mask[:, encoder_hidden_states.shape[1] :] if encoder_hidden_states is not None else mask
            hidden_states = sparse_mlp_compute(
                attn.to_out[0],
                image_mask,
                hidden_states,
                attn.to_out[0].out_features,
            )
            hidden_states = sparse_mlp_compute(
                attn.to_out[1],
                image_mask,
                hidden_states,
                getattr(attn.to_out[1], "out_features", hidden_states.shape[2]),
            )
        else:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class TemporalFlux2KVAttnProcessor(DiffusersFlux2KVAttnProcessor):
    def __call__(
        self,
        attn: DiffusersFlux2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
        spatial_cache: SpatialCache | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return TemporalFlux2AttnProcessor.__call__(
            self,
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            num_ref_tokens=num_ref_tokens,
            spatial_cache=spatial_cache,
            mask=mask,
        )


class TemporalFlux2ParallelSelfAttnProcessor(DiffusersFlux2ParallelSelfAttnProcessor):
    def __call__(
        self,
        attn: DiffusersFlux2ParallelSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        spatial_cache: SpatialCache | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            sparse_mlp_compute(attn.to_qkv_mlp_proj, mask, hidden_states, attn.to_qkv_mlp_proj.out_features)
            if mask is not None
            else attn.to_qkv_mlp_proj(hidden_states)
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "temporal parallel attn module=%s mask_present=%s proj_shape=%s",
                attn.__class__.__name__,
                mask is not None,
                tuple(hidden_states.shape),
            )
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        key, value = _apply_spatial_kv_cache(spatial_cache, mask, key, value, attn, "single")

        hidden_states = sparse_attention_compute(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
            query_mask=mask,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = sparse_mlp_compute(attn.mlp_act_fn, mask, mlp_hidden_states)
        hidden_states = _cat_into_buffer(attn, "_parallel_out_buffer", (hidden_states, mlp_hidden_states), dim=-1)
        hidden_states = sparse_mlp_compute(attn.to_out, mask, hidden_states, attn.to_out.out_features)
        return hidden_states


class TemporalFlux2KVParallelSelfAttnProcessor(DiffusersFlux2KVParallelSelfAttnProcessor):
    def __call__(
        self,
        attn: DiffusersFlux2ParallelSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        kv_cache_mode: str | None = None,
        num_txt_tokens: int = 0,
        num_ref_tokens: int = 0,
        spatial_cache: SpatialCache | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states_proj = (
            sparse_mlp_compute(attn.to_qkv_mlp_proj, mask, hidden_states, attn.to_qkv_mlp_proj.out_features)
            if mask is not None
            else attn.to_qkv_mlp_proj(hidden_states)
        )
        qkv, mlp_hidden_states = torch.split(
            hidden_states_proj, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        key, value = _apply_spatial_kv_cache(spatial_cache, mask, key, value, attn, "single")

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "temporal parallel kv extract stored module=%s ref_range=[%s,%s)",
                    attn.__class__.__name__,
                    ref_start,
                    ref_end,
                )

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "temporal parallel kv cached lookup module=%s num_txt_tokens=%s num_ref_tokens=%s",
                    attn.__class__.__name__,
                    num_txt_tokens,
                    getattr(kv_cache, "num_ref_tokens", None),
                )
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            attn_output = sparse_attention_compute(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
                query_mask=mask,
            )

        attn_output = attn_output.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = sparse_mlp_compute(attn.mlp_act_fn, mask, mlp_hidden_states)
        hidden_states = _cat_into_buffer(attn, "_parallel_out_buffer", (attn_output, mlp_hidden_states), dim=-1)
        hidden_states = sparse_mlp_compute(attn.to_out, mask, hidden_states, attn.to_out.out_features)
        return hidden_states


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
        b, l, _ = qkv.shape
        head_dim = qkv.shape[-1] // (3 * num_heads)
        qkv = qkv.view(b, l, 3, num_heads, head_dim)
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
        if not (qkv.is_cuda and qkv.dtype == torch.float32 and qkv.ndim == 4 and qkv.shape[2] == 3):
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
            return q, k, v
        except Exception:
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
            num_latents_channels=_get_transformer_config(self.pipe).in_channels // 4,
            height=image.shape[-2],
            width=image.shape[-1],
            dtype=dtype,
            device=self.pipe._execution_device,
            generator=generator,
            latents=None,
        )

    def img2img(self, prompt: str, image: PIL.Image.Image, config: Any):
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
            num_latents_channels=_get_transformer_config(pipe).in_channels // 4,
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
        latents = pipe._unpack_latents_with_ids(latents, latent_ids)
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


def prepare_transformer_for_speed(
    pipe: Any,
    *,
    backend: str = "sage",
    fuse_qkv: bool = True,
    patch_klein_ops: bool = True,
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
    if fuse_qkv and hasattr(pipe.transformer, "fuse_qkv_projections"):
        try:
            logger.info("fusing transformer qkv projections")
            pipe.transformer.fuse_qkv_projections()
            setattr(pipe.transformer, "_is_qkv_fused", True)
        except Exception as exc:
            logger.warning("transformer qkv fusion failed, continuing without it: %s", exc)
    try:
        logger.info("fusing double-stream projection stacks")
        fused_blocks = fuse_flux2_double_stream_attention_projections(pipe.transformer)
        if fused_blocks == 0:
            logger.info("no flux2 double-stream projections were fused")
    except Exception as exc:
        logger.warning("double-stream projection fusion failed, continuing without it: %s", exc)
    logger.info("selecting attention backend backend=%s", backend)
    return apply_attention_backend(pipe, backend)


def enable_temporal_consistency(
    pipe: Any,
    *,
    height: int,
    width: int,
    text_seq_len: int = 512,
    output_channels: int = 128,
    attention_head_dim: int = 128,
    num_attention_heads: int = 24,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> SpatialCache:
    """
    Attach temporal-consistency-aware attention processors and return the cache.

    Keep the returned cache alive across frames and pass its mask via
    `TemporalConsistencyController.build_attention_kwargs(...)`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = getattr(pipe.transformer, "dtype", torch.bfloat16)

    spatial_cache = SpatialCache(
        image_seq_len=latent_token_count_for_resolution(pipe, height=height, width=width),
        text_seq_len=text_seq_len,
        output_channels=output_channels,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        device=device,
        dtype=dtype,
    )

    for module in pipe.transformer.modules():
        if isinstance(module, DiffusersFlux2ParallelSelfAttention):
            current = getattr(module, "processor", None)
            if current is None:
                continue
            if current.__class__.__name__.startswith("Flux2KV"):
                new_processor = TemporalFlux2KVParallelSelfAttnProcessor()
            else:
                new_processor = TemporalFlux2ParallelSelfAttnProcessor()
            new_processor._attention_backend = getattr(current, "_attention_backend", None)
            new_processor._parallel_config = getattr(current, "_parallel_config", None)
            module.set_processor(new_processor)
        elif isinstance(module, DiffusersFlux2Attention):
            current = getattr(module, "processor", None)
            if current is None:
                continue
            if current.__class__.__name__.startswith("Flux2KV"):
                new_processor = TemporalFlux2KVAttnProcessor()
            else:
                new_processor = TemporalFlux2AttnProcessor()
            new_processor._attention_backend = getattr(current, "_attention_backend", None)
            new_processor._parallel_config = getattr(current, "_parallel_config", None)
            module.set_processor(new_processor)

    pipe._temporal_spatial_cache = spatial_cache
    return spatial_cache


def _make_fused_linear_from_linears(linears: list[torch.nn.Linear]) -> torch.nn.Linear:
    if not linears:
        raise ValueError("expected at least one linear module")
    first = linears[0]
    if any(l.in_features != first.in_features for l in linears):
        raise ValueError("all fused linears must share input features")
    if any(l.bias is not None for l in linears) and any(l.bias is None for l in linears):
        raise ValueError("all fused linears must either all have bias or all omit bias")
    fused = torch.nn.Linear(
        first.in_features,
        sum(l.out_features for l in linears),
        bias=first.bias is not None,
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    with torch.no_grad():
        fused.weight.copy_(torch.cat([l.weight.detach() for l in linears], dim=0))
        if fused.bias is not None:
            fused.bias.copy_(torch.cat([l.bias.detach() for l in linears], dim=0))
    return fused


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
