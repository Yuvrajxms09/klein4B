"""
KV caching helpers for Flux2 Klein 4B.

Imports helpers from the upstream KV patch so the local pipeline can use the
same attention processors and transformer path without patching Diffusers itself.
"""

from __future__ import annotations

from typing import Any

import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2ParallelSelfAttention,
    Flux2Transformer2DModel,
    _get_qkv_projections,
    dispatch_attention_fn,
    apply_rotary_emb,
)
from diffusers.models.transformers.transformer_flux2 import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale


class Flux2KVLayerCache:
    """Per-layer KV cache for reference image tokens."""

    def __init__(self):
        self.k_ref: torch.Tensor | None = None
        self.v_ref: torch.Tensor | None = None

    def store(self, k_ref: torch.Tensor, v_ref: torch.Tensor):
        self.k_ref = k_ref
        self.v_ref = v_ref

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_ref is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k_ref, self.v_ref

    def clear(self):
        self.k_ref = None
        self.v_ref = None


class Flux2KVCache:
    """Container for all double-stream and single-stream KV layer caches."""

    def __init__(self, num_double_layers: int, num_single_layers: int):
        self.double_block_caches = [Flux2KVLayerCache() for _ in range(num_double_layers)]
        self.single_block_caches = [Flux2KVLayerCache() for _ in range(num_single_layers)]
        self.num_ref_tokens: int = 0

    def get_double(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.double_block_caches[layer_idx]

    def get_single(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.single_block_caches[layer_idx]

    def clear(self):
        for cache in self.double_block_caches:
            cache.clear()
        for cache in self.single_block_caches:
            cache.clear()
        self.num_ref_tokens = 0


def _flux2_kv_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache: Flux2KVLayerCache | None = None,
    backend=None,
) -> torch.Tensor:
    """Causal attention where ref tokens only self-attend and other tokens reuse caches."""
    if num_ref_tokens == 0 and kv_cache is None:
        return dispatch_attention_fn(query, key, value, backend=backend)
    if kv_cache is not None:
        k_ref, v_ref = kv_cache.get()
        k_all = torch.cat([key[:, :num_txt_tokens], k_ref, key[:, num_txt_tokens:]], dim=1)
        v_all = torch.cat([value[:, :num_txt_tokens], v_ref, value[:, num_txt_tokens:]], dim=1)
        return dispatch_attention_fn(query, k_all, v_all, backend=backend)

    ref_start = num_txt_tokens
    ref_end = num_txt_tokens + num_ref_tokens
    q_txt = query[:, :ref_start]
    q_ref = query[:, ref_start:ref_end]
    q_img = query[:, ref_end:]
    k_txt = key[:, :ref_start]
    k_ref = key[:, ref_start:ref_end]
    k_img = key[:, ref_end:]
    v_txt = value[:, :ref_start]
    v_ref = value[:, ref_start:ref_end]
    v_img = value[:, ref_end:]

    q_txt_img = torch.cat([q_txt, q_img], dim=1)
    k_all = torch.cat([k_txt, k_ref, k_img], dim=1)
    v_all = torch.cat([v_txt, v_ref, v_img], dim=1)
    attn_txt_img = dispatch_attention_fn(q_txt_img, k_all, v_all, backend=backend)
    attn_txt = attn_txt_img[:, :ref_start]
    attn_img = attn_txt_img[:, ref_start:]
    attn_ref = dispatch_attention_fn(q_ref, k_ref, v_ref, backend=backend)
    return torch.cat([attn_txt, attn_ref, attn_img], dim=1)


def _blend_mod_params(
    img_params: tuple[torch.Tensor, ...],
    ref_params: tuple[torch.Tensor, ...],
    num_ref: int,
    seq_len: int,
) -> tuple[torch.Tensor, ...]:
    blended = []
    for im, rm in zip(img_params, ref_params):
        if im.ndim == 2:
            im = im.unsqueeze(1)
            rm = rm.unsqueeze(1)
        B = im.shape[0]
        blended.append(
            torch.cat(
                [rm.expand(B, num_ref, -1), im.expand(B, seq_len, -1)[:, num_ref:, :]],
                dim=1,
            )
        )
    return tuple(blended)


def _blend_double_block_mods(
    img_mod: torch.Tensor,
    ref_mod: torch.Tensor,
    num_ref: int,
    seq_len: int,
) -> torch.Tensor:
    if img_mod.ndim == 2:
        img_mod = img_mod.unsqueeze(1)
        ref_mod = ref_mod.unsqueeze(1)
    img_chunks = torch.chunk(img_mod, 6, dim=-1)
    ref_chunks = torch.chunk(ref_mod, 6, dim=-1)
    img_mods = (img_chunks[0:3], img_chunks[3:6])
    ref_mods = (ref_chunks[0:3], ref_chunks[3:6])

    all_params = []
    for img_set, ref_set in zip(img_mods, ref_mods):
        blended = _blend_mod_params(img_set, ref_set, num_ref, seq_len)
        all_params.extend(blended)
    return torch.cat(all_params, dim=-1)


def _blend_single_block_mods(
    single_mod: torch.Tensor,
    ref_mod: torch.Tensor,
    num_txt: int,
    num_ref: int,
    seq_len: int,
) -> torch.Tensor:
    if single_mod.ndim == 2:
        single_mod = single_mod.unsqueeze(1)
        ref_mod = ref_mod.unsqueeze(1)
    img_params = torch.chunk(single_mod, 3, dim=-1)
    ref_params = torch.chunk(ref_mod, 3, dim=-1)

    blended = []
    for im, rm in zip(img_params, ref_params):
        if im.ndim == 2:
            im = im.unsqueeze(1)
            rm = rm.unsqueeze(1)
        B = im.shape[0]
        im_expanded = im.expand(B, seq_len, -1)
        rm_expanded = rm.expand(B, num_ref, -1)
        blended.append(
            torch.cat(
                [im_expanded[:, :num_txt, :], rm_expanded, im_expanded[:, num_txt + num_ref :, :]],
                dim=1,
            )
        )
    return torch.cat(blended, dim=-1)


class Flux2KVAttnProcessor:
    """Attention processor that stores and reuses reference K/V caches."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("Flux2KVAttnProcessor requires torch.scaled_dot_product_attention")

    def __call__(
        self,
        attn: Flux2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        kv_cache: Flux2KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        num_txt_tokens = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class Flux2KVParallelSelfAttnProcessor:
    """KV-aware parallel attention processor for single-stream blocks."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("Flux2KVParallelSelfAttnProcessor requires torch.scaled_dot_product_attention")

    def __call__(
        self,
        attn: Flux2ParallelSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        kv_cache: Flux2KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        num_txt_tokens: int = 0,
        num_ref_tokens: int = 0,
    ) -> torch.Tensor:
        hidden_states_proj = attn.to_qkv_mlp_proj(hidden_states)
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

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            attn_output = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        attn_output = attn_output.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


def set_kv_attn_processors(transformer: Flux2Transformer2DModel):
    """Replace attention processors with KV-aware variants."""
    for block in transformer.transformer_blocks:
        block.attn.set_processor(Flux2KVAttnProcessor())
    for block in transformer.single_transformer_blocks:
        block.attn.set_processor(Flux2KVParallelSelfAttnProcessor())


class Flux2Transformer2DModelKV(Flux2Transformer2DModel):
    """Transformer subclass that exposes KV cache arguments."""

    _skip_keys = ["kv_cache"]

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        kv_cache: Flux2KVCache | None = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
        ref_fixed_timestep: float = 0.0,
    ) -> tuple[torch.Tensor, Flux2KVCache] | Transformer2DModelOutput:
        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            num_img_tokens = hidden_states.shape[1]
            kv_cache = Flux2KVCache(
                num_double_layers=len(self.transformer_blocks),
                num_single_layers=len(self.single_transformer_blocks),
            )
            kv_cache.num_ref_tokens = num_ref_tokens
            ref_timestep = torch.full_like(timestep, ref_fixed_timestep * 1000)
            ref_temb = self.time_guidance_embed(ref_timestep, guidance)
            ref_double_mod_img = self.double_stream_modulation_img(ref_temb)
            ref_single_mod = self.single_stream_modulation(ref_temb)
            double_stream_mod_img = _blend_double_block_mods(
                double_stream_mod_img, ref_double_mod_img, num_ref_tokens, num_img_tokens
            )
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        if kv_cache_mode == "extract":
            kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "extract",
                "num_ref_tokens": num_ref_tokens,
            }
        elif kv_cache_mode == "cached" and kv_cache is not None:
            kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "cached",
                "num_ref_tokens": kv_cache.num_ref_tokens,
            }
        else:
            kv_attn_kwargs = joint_attention_kwargs

        for index_block, block in enumerate(self.transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs["kv_cache"] = kv_cache.get_double(index_block)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    kv_attn_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=kv_attn_kwargs,
                )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            total_single_len = hidden_states.shape[1]
            single_stream_mod = _blend_single_block_mods(
                single_stream_mod, ref_single_mod, num_txt_tokens, num_ref_tokens, total_single_len
            )
        if kv_cache_mode is not None:
            kv_attn_kwargs_single = {**kv_attn_kwargs, "num_txt_tokens": num_txt_tokens}
        else:
            kv_attn_kwargs_single = kv_attn_kwargs

        for index_block, block in enumerate(self.single_transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs_single["kv_cache"] = kv_cache.get_single(index_block)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    kv_attn_kwargs_single,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=kv_attn_kwargs_single,
                )
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = hidden_states[:, num_txt_tokens + num_ref_tokens :, ...]
        else:
            hidden_states = hidden_states[:, num_txt_tokens:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if kv_cache_mode == "extract":
            if not return_dict:
                return (output,), kv_cache
            return Transformer2DModelOutput(sample=output), kv_cache

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
