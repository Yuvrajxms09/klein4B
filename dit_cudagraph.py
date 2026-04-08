# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Manual CUDA Graph capture for a single Flux2 Klein DiT forward (cond and optional uncond)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiTGraphKey:
    """Invalidates captured graphs when any of these change."""

    device_index: int
    batch: int
    latent_seq: int
    packed_seq: int
    latent_channels: int
    text_seq: int
    text_width: int
    latent_dtype: torch.dtype
    compute_dtype: torch.dtype
    needs_uncond_graph: bool


class KleinDiTCudaGraphRunner:
    """
    Owns static I/O tensors and one or two CUDAGraphs mirroring the Klein pipeline DiT calls.

    Replay copies fresh latents / timestep / text into static slots, then runs graph.replay().
    Encoder and RoPE ids are copied each step so prompt/callback updates stay correct.
    """

    __slots__ = (
        "_key",
        "_stream",
        "_graph_cond",
        "_graph_uncond",
        "_static_hidden",
        "_static_timestep",
        "_static_encoder_cond",
        "_static_txt_cond",
        "_static_encoder_uncond",
        "_static_txt_uncond",
        "_static_img_ids",
        "_sample_cond",
        "_sample_uncond",
        "_latent_seq",
    )

    def __init__(self) -> None:
        self._key: DiTGraphKey | None = None
        # Created in try_capture on the same device as latents (not at pipeline __init__, which may be CPU).
        self._stream: torch.cuda.Stream | None = None
        self._graph_cond: torch.cuda.CUDAGraph | None = None
        self._graph_uncond: torch.cuda.CUDAGraph | None = None
        self._static_hidden: torch.Tensor | None = None
        self._static_timestep: torch.Tensor | None = None
        self._static_encoder_cond: torch.Tensor | None = None
        self._static_txt_cond: torch.Tensor | None = None
        self._static_encoder_uncond: torch.Tensor | None = None
        self._static_txt_uncond: torch.Tensor | None = None
        self._static_img_ids: torch.Tensor | None = None
        self._sample_cond: torch.Tensor | None = None
        self._sample_uncond: torch.Tensor | None = None
        self._latent_seq: int = 0

    def reset(self) -> None:
        self._key = None
        self._graph_cond = None
        self._graph_uncond = None
        self._static_hidden = None
        self._static_timestep = None
        self._static_encoder_cond = None
        self._static_txt_cond = None
        self._static_encoder_uncond = None
        self._static_txt_uncond = None
        self._static_img_ids = None
        self._sample_cond = None
        self._sample_uncond = None
        self._latent_seq = 0
        self._stream = None

    @property
    def is_ready(self) -> bool:
        return self._graph_cond is not None

    def key_matches(self, key: DiTGraphKey) -> bool:
        return self._key == key and self.is_ready

    def try_capture(
        self,
        *,
        key: DiTGraphKey,
        transformer: Any,
        latent_seq: int,
        joint_attention_kwargs: dict[str, Any] | None,
        timesteps_0: torch.Tensor,
        latents: torch.Tensor,
        image_latents: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_text_ids: torch.Tensor | None,
        latent_image_ids: torch.Tensor,
    ) -> bool:
        self.reset()
        self._latent_seq = latent_seq
        self._key = key

        device = latents.device
        self._stream = torch.cuda.Stream(device=device)
        dt = transformer.dtype
        B, P, C = key.batch, key.packed_seq, key.latent_channels

        try:
            self._static_hidden = torch.empty((B, P, C), device=device, dtype=dt)
            self._static_timestep = torch.empty((B,), device=device, dtype=latents.dtype)
            self._static_encoder_cond = torch.empty(
                (B, key.text_seq, key.text_width),
                device=device,
                dtype=prompt_embeds.dtype,
            )
            self._static_txt_cond = torch.empty_like(text_ids)
            self._static_img_ids = torch.empty_like(latent_image_ids)

            if key.needs_uncond_graph:
                if negative_prompt_embeds is None or negative_text_ids is None:
                    raise ValueError("CFG graph requested but negative embeddings are missing")
                self._static_encoder_uncond = torch.empty_like(negative_prompt_embeds)
                self._static_txt_uncond = torch.empty_like(negative_text_ids)
            else:
                self._static_encoder_uncond = None
                self._static_txt_uncond = None

            self._pack_static_hidden(latents, image_latents, dt)
            self._copy_timestep(timesteps_0, latents)
            self._static_encoder_cond.copy_(prompt_embeds)
            self._static_txt_cond.copy_(text_ids)
            self._static_img_ids.copy_(latent_image_ids)

            if key.needs_uncond_graph:
                self._static_encoder_uncond.copy_(negative_prompt_embeds)
                self._static_txt_uncond.copy_(negative_text_ids)

            s = self._stream
            wf = torch.cuda.current_stream(device)

            def forward_cond() -> torch.Tensor:
                with transformer.cache_context("cond"):
                    return transformer(
                        hidden_states=self._static_hidden,
                        timestep=self._static_timestep,
                        guidance=None,
                        encoder_hidden_states=self._static_encoder_cond,
                        txt_ids=self._static_txt_cond,
                        img_ids=self._static_img_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]

            for _ in range(3):
                with torch.cuda.stream(s):
                    _ = forward_cond()
                wf.wait_stream(s)
            torch.cuda.synchronize(device=device)

            g_cond = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_cond, stream=s):
                with transformer.cache_context("cond"):
                    self._sample_cond = transformer(
                        hidden_states=self._static_hidden,
                        timestep=self._static_timestep,
                        guidance=None,
                        encoder_hidden_states=self._static_encoder_cond,
                        txt_ids=self._static_txt_cond,
                        img_ids=self._static_img_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]

            self._graph_cond = g_cond

            if key.needs_uncond_graph:

                def forward_uncond() -> torch.Tensor:
                    with transformer.cache_context("uncond"):
                        return transformer(
                            hidden_states=self._static_hidden,
                            timestep=self._static_timestep,
                            guidance=None,
                            encoder_hidden_states=self._static_encoder_uncond,
                            txt_ids=self._static_txt_uncond,
                            img_ids=self._static_img_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]

                for _ in range(3):
                    with torch.cuda.stream(s):
                        _ = forward_uncond()
                    wf.wait_stream(s)
                torch.cuda.synchronize(device=device)

                g_uncond = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_uncond, stream=s):
                    with transformer.cache_context("uncond"):
                        self._sample_uncond = transformer(
                            hidden_states=self._static_hidden,
                            timestep=self._static_timestep,
                            guidance=None,
                            encoder_hidden_states=self._static_encoder_uncond,
                            txt_ids=self._static_txt_uncond,
                            img_ids=self._static_img_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]

                self._graph_uncond = g_uncond
            else:
                self._graph_uncond = None
                self._sample_uncond = None

        except Exception as exc:
            logger.warning("DiT CUDA graph capture failed, using eager DiT: %s", exc)
            self.reset()
            return False

        logger.info(
            "DiT CUDA graph ready: packed_seq=%d latent_seq=%d cfg_graphs=%s",
            P,
            latent_seq,
            key.needs_uncond_graph,
        )
        return True

    def _pack_static_hidden(
        self,
        latents: torch.Tensor,
        image_latents: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> None:
        L = latents.size(1)
        l_eff = latents.to(compute_dtype) if latents.dtype != compute_dtype else latents
        if image_latents is None:
            self._static_hidden.copy_(l_eff)
        else:
            i_eff = image_latents.to(compute_dtype) if image_latents.dtype != compute_dtype else image_latents
            self._static_hidden[:, :L].copy_(l_eff)
            self._static_hidden[:, L:].copy_(i_eff)

    def _copy_timestep(self, t_step: torch.Tensor, latents: torch.Tensor) -> None:
        ts = t_step.expand(latents.shape[0]).to(latents.dtype)
        self._static_timestep.copy_(ts / 1000)

    def _replay_stream_sync(self, device: torch.device) -> None:
        s = self._stream
        if s is None:
            raise RuntimeError("DiT CUDA graph stream is not initialized")
        wf = torch.cuda.current_stream(device)
        wf.wait_stream(s)

    def replay_cond(
        self,
        *,
        device: torch.device,
        latents: torch.Tensor,
        image_latents: torch.Tensor | None,
        t_step: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        dt = compute_dtype
        s = self._stream
        if s is None:
            raise RuntimeError("DiT CUDA graph replay before successful capture")
        with torch.cuda.stream(s):
            self._pack_static_hidden(latents, image_latents, dt)
            self._copy_timestep(t_step, latents)
            self._static_encoder_cond.copy_(prompt_embeds)
            self._static_txt_cond.copy_(text_ids)
            self._static_img_ids.copy_(latent_image_ids)
            self._graph_cond.replay()
        self._replay_stream_sync(device)
        return self._sample_cond[:, : self._latent_seq]

    def replay_uncond(
        self,
        *,
        device: torch.device,
        latents: torch.Tensor,
        image_latents: torch.Tensor | None,
        t_step: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        dt = compute_dtype
        s = self._stream
        if s is None:
            raise RuntimeError("DiT CUDA graph replay before successful capture")
        with torch.cuda.stream(s):
            self._pack_static_hidden(latents, image_latents, dt)
            self._copy_timestep(t_step, latents)
            self._static_encoder_uncond.copy_(negative_prompt_embeds)
            self._static_txt_uncond.copy_(negative_text_ids)
            self._static_img_ids.copy_(latent_image_ids)
            self._graph_uncond.replay()
        self._replay_stream_sync(device)
        return self._sample_uncond[:, : self._latent_seq]
