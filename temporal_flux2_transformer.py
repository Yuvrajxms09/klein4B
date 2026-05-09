# Copyright 2025 Black Forest Labs, The HuggingFace Team, TensorForger (FluxRT), and this file's authors.
# SPDX-License-Identifier: Apache-2.0
#
# FluxRT-aligned temporal paths for FLUX.2 Klein when used with diffusers:
# - Double-stream block: masked sparse FFN (image + text), matching FluxRT Flux2TransformerBlock.
# - Model tail: masked sparse proj_out + spatial output cache, matching FluxRT Flux2Transformer2DModel.
#
# See FluxRT: src/fluxrt/stream_processor/transformer_flux2.py (FluxRT upstream).

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
    Flux2TransformerBlock,
    Flux2Modulation,
    _blend_double_block_mods,
    _blend_single_block_mods,
)
from diffusers.utils import apply_lora_scale

from cache_dit_klein import Flux2KVCache, sparse_mlp_compute


class TemporalFlux2TransformerBlock(Flux2TransformerBlock):
    """FluxRT-equivalent double-stream block: sparse FFN on text + image tokens when `mask` is present."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        mask = joint_attention_kwargs.get("mask", None)

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(
            temb_mod_img, 2
        )
        (
            (c_shift_msa, c_scale_msa, c_gate_msa),
            (c_shift_mlp, c_scale_mlp, c_gate_mlp),
        ) = Flux2Modulation.split(temb_mod_txt, 2)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs

        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        text_seq_len = encoder_hidden_states.shape[1]
        if mask is not None:
            ff_output = sparse_mlp_compute(
                self.ff,
                mask[:, text_seq_len:],
                norm_hidden_states,
                norm_hidden_states.shape[2],
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_mlp * ff_output

        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        )

        if mask is not None:
            context_ff_output = sparse_mlp_compute(
                self.ff_context,
                mask[:, :text_seq_len],
                norm_encoder_hidden_states,
                norm_encoder_hidden_states.shape[2],
            )
        else:
            context_ff_output = self.ff_context(norm_encoder_hidden_states)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


def _double_block_to_temporal(block: Flux2TransformerBlock) -> TemporalFlux2TransformerBlock:
    if isinstance(block, TemporalFlux2TransformerBlock):
        return block
    # Keep the existing module instance so weights, fused projections, and
    # attention backend settings survive the temporal conversion.
    block.__class__ = TemporalFlux2TransformerBlock
    return block


def replace_double_stream_blocks_with_temporal(transformer: Flux2Transformer2DModel) -> None:
    """In-place: swap each double-stream block for TemporalFlux2TransformerBlock (same weights)."""
    for index, block in enumerate(transformer.transformer_blocks):
        transformer.transformer_blocks[index] = _double_block_to_temporal(block)


class TemporalFlux2Transformer2DModel(Flux2Transformer2DModel):
    """
    Diffusers Flux2Transformer2DModel with FluxRT spatial-cache behavior:
    - preprocess_mask at forward entry (mask + spatial_cache from kwargs or top-level args)
    - block_id / mask / spatial_cache threaded like FluxRT
    - sparse proj_out + sync_with_output_cache on the tail
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        replace_double_stream_blocks_with_temporal(self)

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
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
        spatial_cache: Any | None = None,
        mask: torch.Tensor | None = None,
        **unused_kwargs: Any,
    ) -> torch.Tensor | Flux2Transformer2DModelOutput:
        del unused_kwargs
        joint_attention_kwargs = dict(joint_attention_kwargs or {})
        if spatial_cache is not None:
            joint_attention_kwargs.setdefault("spatial_cache", spatial_cache)
        if mask is not None:
            joint_attention_kwargs.setdefault("mask", mask)

        mask = joint_attention_kwargs.get("mask")
        spatial_cache = joint_attention_kwargs.get("spatial_cache")

        # FluxRT: preprocess runs whenever mask is set; requires spatial_cache (same contract as FluxRT pipeline).
        if mask is not None:
            if spatial_cache is None:
                raise ValueError("spatial_cache is required when mask is passed (FluxRT temporal consistency path)")
            mask = spatial_cache.preprocess_mask(mask)
            joint_attention_kwargs["mask"] = mask

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
                double_stream_mod_img,
                ref_double_mod_img,
                num_ref_tokens,
                num_img_tokens,
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
                **joint_attention_kwargs,
                "kv_cache": None,
                "kv_cache_mode": "extract",
                "num_ref_tokens": num_ref_tokens,
            }
        elif kv_cache_mode == "cached" and kv_cache is not None:
            kv_attn_kwargs = {
                **joint_attention_kwargs,
                "kv_cache": None,
                "kv_cache_mode": "cached",
                "num_ref_tokens": kv_cache.num_ref_tokens,
            }
        else:
            kv_attn_kwargs = joint_attention_kwargs

        for index_block, block in enumerate(self.transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs["kv_cache"] = kv_cache.get_double(index_block)

            kv_attn_kwargs["block_id"] = index_block
            kv_attn_kwargs["spatial_cache"] = spatial_cache
            kv_attn_kwargs["mask"] = mask

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
                single_stream_mod,
                ref_single_mod,
                num_txt_tokens,
                num_ref_tokens,
                total_single_len,
            )

        if kv_cache_mode is not None:
            kv_attn_kwargs_single = {**kv_attn_kwargs, "num_txt_tokens": num_txt_tokens}
        else:
            kv_attn_kwargs_single = kv_attn_kwargs

        for index_block, block in enumerate(self.single_transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs_single["kv_cache"] = kv_cache.get_single(index_block)

            # Single-stream blocks read ``joint_attention_kwargs_single``; keep block_id / cache in sync each step.
            kv_attn_kwargs_single["block_id"] = index_block
            kv_attn_kwargs_single["spatial_cache"] = spatial_cache
            kv_attn_kwargs_single["mask"] = mask

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
        if mask is not None:
            # Match FluxRT: slice from embedded text length (same as num_txt_tokens here).
            output = sparse_mlp_compute(
                self.proj_out,
                mask[:, encoder_hidden_states.shape[1] :],
                hidden_states,
                self.proj_out.out_features,
            )
        else:
            output = self.proj_out(hidden_states)

        if spatial_cache is not None:
            output = spatial_cache.sync_with_output_cache(mask, output)

        if kv_cache_mode == "extract":
            if not return_dict:
                return (output, kv_cache)
            return Flux2Transformer2DModelOutput(sample=output, kv_cache=kv_cache)

        if not return_dict:
            return (output,)

        return Flux2Transformer2DModelOutput(sample=output)


def convert_transformer_for_temporal(
    transformer: Flux2Transformer2DModel,
) -> TemporalFlux2Transformer2DModel:
    """
    Build a TemporalFlux2Transformer2DModel from an existing diffusers transformer (same config/weights).

    Call after ``from_pretrained`` and **before** ``torch.compile`` on the transformer unless you re-compile
    the converted module yourself.
    """
    if isinstance(transformer, TemporalFlux2Transformer2DModel):
        replace_double_stream_blocks_with_temporal(transformer)
        return transformer

    transformer.__class__ = TemporalFlux2Transformer2DModel
    replace_double_stream_blocks_with_temporal(transformer)
    return transformer
