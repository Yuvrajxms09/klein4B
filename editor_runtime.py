from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Literal
from pathlib import Path

import numpy as np

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle


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
            clipped = np.clip(image, 0.0, 1.0)
            return (clipped * 255).astype(np.uint8)
        return image.astype(np.uint8)

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

    def set_prompt(self, prompt: str) -> None:
        self.stream_processor.set_prompt(prompt)

    def set_reference_image(self, image, *, colorspace: ColorSpace = "rgb") -> None:
        frame = self._as_numpy(image)
        frame = self._ensure_uint8(frame)
        frame = self._to_rgb(frame, colorspace)
        frame = crop_maximal_rectangle(frame, self._resolution["height"], self._resolution["width"])
        self.stream_processor.set_reference_image(frame)

    def set_canvas(self, image, *, colorspace: ColorSpace = "rgb") -> None:
        frame = self._as_numpy(image)
        frame = self._ensure_uint8(frame)
        frame = self._to_bgr(frame, colorspace)
        frame = crop_maximal_rectangle(frame, self._resolution["height"], self._resolution["width"])
        self.input_tensor.copy_from(frame)

    def set_mask(self, mask, *, full_resolution: bool = True) -> None:
        mask_arr = self._as_numpy(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]

        if mask_arr.dtype != np.uint8:
            if np.issubdtype(mask_arr.dtype, np.floating):
                mask_arr = np.clip(mask_arr, 0.0, 1.0)
                mask_arr = (mask_arr * 255).astype(np.uint8)
            else:
                mask_arr = mask_arr.astype(np.uint8)

        if full_resolution:
            import cv2

            latent_h = self._resolution["height"] // 16
            latent_w = self._resolution["width"] // 16
            mask_arr = cv2.resize(mask_arr, (latent_w, latent_h), interpolation=cv2.INTER_NEAREST)

        mask_arr = (mask_arr > 0).astype(np.uint8) * 2
        self.stream_processor.set_mask(mask_arr)

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
    ) -> None:
        if prompt is not None:
            self.set_prompt(prompt)
        if reference_image is not None:
            self.set_reference_image(reference_image, colorspace=reference_colorspace)
        if canvas is not None:
            self.set_canvas(canvas, colorspace=canvas_colorspace)
        if mask is not None:
            self.set_mask(mask, full_resolution=full_resolution_mask)

    def get_latest_frame(self, *, colorspace: ColorSpace = "rgb") -> np.ndarray:
        with self._lock:
            frame = self.output_tensor.to_numpy()
        return self._to_rgb(frame, "bgr") if colorspace == "rgb" else frame

    def get_shared_tensors(self):
        return self.input_tensor, self.output_tensor
