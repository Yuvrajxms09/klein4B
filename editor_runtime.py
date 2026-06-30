from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Literal
from pathlib import Path

import cv2
import numpy as np

from fluxrt import StreamProcessor
from fluxrt.utils.crop_maximal_rectangle import crop_maximal_rectangle


ColorSpace = Literal["rgb", "bgr"]


@dataclass(frozen=True)
class EditorFrame:
    image: np.ndarray
    colorspace: ColorSpace


class FluxRTEditorRuntime:
    """
    Thin editor-facing wrapper around the transplanted FluxRT StreamProcessor.

    The runtime keeps the FluxRT engine intact and only handles editor concerns:
    - canvas frame color-space normalization
    - aspect-ratio preserving resize/crop to the model resolution
    - manual mask downsampling to the latent grid
    - prompt / reference-image forwarding
    """

    def __init__(self, config_path: str, start: bool = True):
        self.stream_processor = StreamProcessor(self._resolve_config_path(config_path))
        self._lock = threading.RLock()
        self._started = False
        self._resolution = self.stream_processor.get_resolution()
        self._reference_resolution = self.stream_processor.config.get(
            "reference_image_resolution", self._resolution
        )

        self.input_tensor = self.stream_processor.get_input_tensor()
        self.output_tensor = self.stream_processor.get_output_tensor()

        if start:
            self.start()

    @property
    def resolution(self) -> dict:
        return self._resolution

    @staticmethod
    def _resolve_config_path(config_path: str) -> str:
        path = Path(config_path)
        if path.is_absolute():
            return str(path)

        repo_root = Path(__file__).resolve().parent
        candidate = (repo_root / path).resolve()
        if candidate.exists():
            return str(candidate)
        return str(path.resolve())

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self.stream_processor.start()
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self.stream_processor.stop()
            self._started = False

    def wait_until_ready(self, timeout_s: float | None = None) -> bool:
        import time

        deadline = None if timeout_s is None else time.time() + timeout_s
        while True:
            if self.stream_processor.is_ready():
                return True
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(0.01)

    @staticmethod
    def _as_numpy(image) -> np.ndarray:
        if image is None:
            raise ValueError("image is required")
        if isinstance(image, np.ndarray):
            return image
        if hasattr(image, "__array__"):
            return np.asarray(image)
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    @staticmethod
    def _ensure_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        if np.issubdtype(image.dtype, np.floating):
            min_value = float(np.nanmin(image))
            max_value = float(np.nanmax(image))
            if -1.0 <= min_value and max_value <= 1.0:
                if min_value < 0.0:
                    clipped = np.clip(image, -1.0, 1.0)
                    return (((clipped + 1.0) * 0.5) * 255.0).astype(np.uint8)
                clipped = np.clip(image, 0.0, 1.0)
                return (clipped * 255).astype(np.uint8)
            return np.clip(image, 0.0, 255.0).astype(np.uint8)
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_bgr(image: np.ndarray, colorspace: ColorSpace) -> np.ndarray:
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if colorspace == "bgr":
            return image
        return image[..., ::-1]

    @staticmethod
    def _to_rgb(image: np.ndarray, colorspace: ColorSpace) -> np.ndarray:
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if colorspace == "rgb":
            return image
        return image[..., ::-1]

    @staticmethod
    def _fit_to_resolution(
        image: np.ndarray,
        *,
        height: int,
        width: int,
        interpolation: int,
    ) -> np.ndarray:
        aspect_ratio = width / height
        input_height, input_width = image.shape[:2]
        input_aspect_ratio = input_width / input_height

        if aspect_ratio > input_aspect_ratio:
            crop_height = int(round(input_width / aspect_ratio))
            crop_height = max(1, min(input_height, crop_height))
            crop_width = input_width
        else:
            crop_width = int(round(input_height * aspect_ratio))
            crop_width = max(1, min(input_width, crop_width))
            crop_height = input_height

        start_x = (input_width - crop_width) // 2
        start_y = (input_height - crop_height) // 2
        cropped = image[start_y : start_y + crop_height, start_x : start_x + crop_width]
        return cv2.resize(cropped, (width, height), interpolation=interpolation)

    def set_prompt(self, prompt: str) -> None:
        self.stream_processor.set_prompt(prompt)

    def set_reference_image(self, image, *, colorspace: ColorSpace = "rgb") -> None:
        frame = self._as_numpy(image)
        frame = self._ensure_uint8(frame)
        frame = self._to_rgb(frame, colorspace)
        frame = crop_maximal_rectangle(
            frame,
            self._reference_resolution["height"],
            self._reference_resolution["width"],
        )
        self.stream_processor.set_reference_image(frame)

    def clear_reference_image(self) -> None:
        self.stream_processor.set_reference_image(None)

    def set_canvas(self, image, *, colorspace: ColorSpace = "rgb") -> None:
        frame = self._as_numpy(image)
        frame = self._ensure_uint8(frame)
        frame = self._to_bgr(frame, colorspace)
        frame = crop_maximal_rectangle(
            frame, self._resolution["height"], self._resolution["width"]
        )
        self.input_tensor.copy_from(frame)

    def set_mask(self, mask, *, full_resolution: bool = True) -> None:
        mask_arr = self._as_numpy(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        if mask_arr.ndim != 2:
            raise ValueError(
                f"mask must be 2D after channel squeeze, got shape {mask_arr.shape}"
            )

        if mask_arr.dtype != np.uint8:
            if np.issubdtype(mask_arr.dtype, np.floating):
                if float(np.nanmax(mask_arr)) <= 1.0:
                    mask_arr = np.clip(mask_arr, 0.0, 1.0)
                    mask_arr = (mask_arr * 255).astype(np.uint8)
                else:
                    mask_arr = np.clip(mask_arr, 0.0, 255.0).astype(np.uint8)
            else:
                mask_arr = np.clip(mask_arr, 0, 255).astype(np.uint8)

        if full_resolution:
            latent_h = self._resolution["height"] // 16
            latent_w = self._resolution["width"] // 16
            mask_arr = self._fit_to_resolution(
                mask_arr,
                height=latent_h,
                width=latent_w,
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            expected_shape = (
                self._resolution["height"] // 16,
                self._resolution["width"] // 16,
            )
            if mask_arr.shape != expected_shape:
                raise ValueError(
                    f"latent mask must have shape {expected_shape}, got {mask_arr.shape}"
                )

        mask_arr = (mask_arr > 0).astype(np.uint8) * 2
        self.stream_processor.set_mask(mask_arr)

    def clear_mask(self) -> None:
        latent_h = self._resolution["height"] // 16
        latent_w = self._resolution["width"] // 16
        self.stream_processor.set_mask(np.zeros((latent_h, latent_w), dtype=np.uint8))

    def update(
        self,
        *,
        prompt: str | None = None,
        canvas=None,
        mask=None,
        reference_image=None,
        canvas_colorspace: ColorSpace = "rgb",
        reference_colorspace: ColorSpace = "rgb",
        full_resolution_mask: bool = True,
        clear_reference_image: bool = False,
        clear_mask: bool = False,
    ) -> None:
        if prompt is not None:
            self.set_prompt(prompt)
        if clear_reference_image:
            self.clear_reference_image()
        elif reference_image is not None:
            self.set_reference_image(reference_image, colorspace=reference_colorspace)
        if canvas is not None:
            self.set_canvas(canvas, colorspace=canvas_colorspace)
        if clear_mask:
            self.clear_mask()
        elif mask is not None:
            self.set_mask(mask, full_resolution=full_resolution_mask)

    def get_latest_frame(self, *, colorspace: ColorSpace = "rgb") -> np.ndarray:
        with self._lock:
            frame = self.output_tensor.to_numpy()
        return self._to_rgb(frame, "bgr") if colorspace == "rgb" else frame

    def get_shared_tensors(self):
        return self.input_tensor, self.output_tensor
