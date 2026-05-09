from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import logging
import time
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass
class TemporalConsistencyConfig:
    height: int
    width: int
    compression_ratio: int = 16
    text_seq_len: int = 512
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    reset_period: float | None = None
    always_update_image_cache: bool = True
    mask_calculation_method: str = "auto"
    reference_image_seq_len: int | None = None
    debug: bool = False


class TemporalConsistencyController:
    """
    Frame-difference controller for webcam/video v2v use cases.

    Produces a token mask compatible with the spatial cache path in `cache_dit_klein.py`.
    Mask values:
    - 0: skip
    - 1: execute only
    - 2: execute and update
    """

    def __init__(self, config: TemporalConsistencyConfig):
        self.config = config
        self.height = int(config.height)
        self.width = int(config.width)
        self.compression_ratio = int(config.compression_ratio)
        self.text_seq_len = int(config.text_seq_len)
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.reference_image_seq_len = config.reference_image_seq_len

        self.mask_height = self.height // self.compression_ratio
        self.mask_width = self.width // self.compression_ratio

        self.cached_frame = torch.zeros(1, 3, self.height, self.width, device=self.device, dtype=self.dtype)
        self.reset_period = config.reset_period
        self.always_update_image_cache = config.always_update_image_cache
        self.mask_calculation_method = config.mask_calculation_method
        self.debug = bool(config.debug)

        self.requires_reset = False
        self.requires_update_image_cache = True
        self.text_is_valid = False
        self.reference_image_is_valid = False
        self.previous_reset = time.time() if self.reset_period is not None else 0.0

        self.manual_mask: torch.Tensor | None = None

        logger.info(
            "temporal controller init height=%s width=%s compression_ratio=%s text_seq_len=%s ref_seq_len=%s reset_period=%s manual=%s debug=%s",
            self.height,
            self.width,
            self.compression_ratio,
            self.text_seq_len,
            self.reference_image_seq_len,
            self.reset_period,
            self.mask_calculation_method == "manual",
            self.debug,
        )

    @staticmethod
    def _to_bchw(frame: torch.Tensor | np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame)
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = frame.unsqueeze(0)
        elif frame.ndim == 3 and frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1).unsqueeze(0)
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError("frame must be [1, 3, H, W], [3, H, W], or [H, W, 3]")
        return frame.to(device=device, dtype=dtype)

    def set_mask(self, mask: torch.Tensor | np.ndarray) -> None:
        if self.mask_calculation_method != "manual":
            raise ValueError("Mask calculation method is not set to manual.")
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        self.manual_mask = mask.to(device=self.device, dtype=torch.int32)
        if self.debug or logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "temporal controller manual mask set shape=%s active=%s",
                tuple(self.manual_mask.shape),
                int((self.manual_mask != 0).sum().item()),
            )

    def update_image_cache(self) -> None:
        self.requires_update_image_cache = True
        if self.debug or logger.isEnabledFor(logging.DEBUG):
            logger.debug("temporal controller image cache marked dirty")

    def reset_cache(self) -> None:
        self.requires_reset = True
        self.text_is_valid = False
        self.reference_image_is_valid = False
        if self.reset_period is not None:
            self.previous_reset = time.time()
        logger.info("temporal controller cache reset requested")

    @staticmethod
    def _mask_counts(mask: torch.Tensor) -> dict[str, int]:
        return {
            "skip": int((mask == 0).sum().item()),
            "execute": int((mask == 1).sum().item()),
            "update": int((mask == 2).sum().item()),
        }

    def update_and_get_mask(self, frame: torch.Tensor | np.ndarray) -> torch.Tensor:
        if self.mask_calculation_method == "manual":
            if self.manual_mask is None:
                logger.warning("temporal controller manual mode requested but no mask was set; returning zeros")
                return torch.zeros(
                    1, self.mask_height, self.mask_width, device=self.device, dtype=torch.int32
                )
            if self.debug or logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "temporal controller manual mask used shape=%s counts=%s",
                    tuple(self.manual_mask.shape),
                    self._mask_counts(self.manual_mask),
                )
            return self.manual_mask

        frame = self._to_bchw(frame, self.device, self.dtype)
        if self.debug or logger.isEnabledFor(logging.DEBUG):
            logger.debug("temporal controller frame received shape=%s dtype=%s device=%s", tuple(frame.shape), frame.dtype, frame.device)

        if self.reset_period is not None and time.time() - float(self.previous_reset) > float(self.reset_period):
            self.requires_reset = True
            logger.info("temporal controller reset period elapsed reset_period=%s", self.reset_period)

        if self.requires_reset:
            self.cached_frame = frame
            self.previous_reset = time.time()
            self.requires_reset = False
            logger.info("temporal controller hard reset applied frame cached and mask forced to update")
            return torch.full(
                (1, self.mask_height, self.mask_width), 2, device=self.device, dtype=torch.int32
            )

        frame_small = F.avg_pool2d(frame, kernel_size=3, stride=1, padding=1)
        cached_small = F.avg_pool2d(self.cached_frame, kernel_size=3, stride=1, padding=1)
        difference = (cached_small - frame_small).pow(2).mean(dim=1, keepdim=True)

        difference_mask = F.max_pool2d(difference, (self.compression_ratio, self.compression_ratio))
        difference_mask = difference_mask > 0.1
        difference_mask = F.max_pool2d(difference_mask.float(), kernel_size=3, stride=1, padding=1) > 0
        difference_mask = F.max_pool2d(difference_mask.float(), kernel_size=3, stride=1, padding=1) > 0

        if self.debug or logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "temporal controller diff stats mean=%.6f max=%.6f threshold=0.1",
                float(difference.mean().item()),
                float(difference.max().item()),
            )

        difference_mask_up = F.interpolate(
            difference_mask.float(), size=(self.height, self.width), mode="nearest"
        )
        self.cached_frame = torch.where(
            difference_mask_up.to(torch.bool).expand(-1, 3, -1, -1),
            frame,
            self.cached_frame,
        )

        image_mask = difference_mask.squeeze(1).to(torch.int32)
        if self.requires_update_image_cache:
            if not self.always_update_image_cache:
                self.requires_update_image_cache = False
            logger.info(
                "temporal controller forcing image-cache update counts=%s",
                self._mask_counts(image_mask * 2),
            )
            return image_mask * 2
        if self.debug or logger.isEnabledFor(logging.DEBUG):
            logger.debug("temporal controller computed image mask counts=%s", self._mask_counts(image_mask))
        return image_mask

    def use_text_mask(self) -> torch.Tensor:
        if self.text_is_valid:
            mask = torch.zeros(1, self.text_seq_len, device=self.device, dtype=torch.int32)
        else:
            mask = torch.full((1, self.text_seq_len), 2, device=self.device, dtype=torch.int32)
            self.text_is_valid = True
        return mask

    def use_reference_image_mask(self) -> torch.Tensor | None:
        if self.reference_image_seq_len is None:
            return None
        if self.reference_image_is_valid:
            mask = torch.zeros(1, self.reference_image_seq_len, device=self.device, dtype=torch.int32)
        else:
            mask = torch.full((1, self.reference_image_seq_len), 2, device=self.device, dtype=torch.int32)
            self.reference_image_is_valid = True
        return mask

    def build_attention_kwargs(
        self,
        frame: torch.Tensor | np.ndarray,
        *,
        spatial_cache: Any | None = None,
    ) -> dict[str, Any]:
        image_mask = self.update_and_get_mask(frame).reshape(1, -1)
        mask = torch.cat([self.use_text_mask(), image_mask, image_mask], dim=-1)
        ref_mask = self.use_reference_image_mask()
        if ref_mask is not None:
            mask = torch.cat([mask, ref_mask], dim=-1)
        if spatial_cache is not None:
            mask = spatial_cache.preprocess_mask(mask)
        kwargs: dict[str, Any] = {"mask": mask}
        if spatial_cache is not None:
            kwargs["spatial_cache"] = spatial_cache
        kwargs["temporal_controller"] = self
        if self.debug or logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "temporal controller attention kwargs built mask_shape=%s counts=%s has_spatial_cache=%s has_ref_mask=%s",
                tuple(mask.shape),
                self._mask_counts(mask),
                spatial_cache is not None,
                ref_mask is not None,
            )
        return kwargs
