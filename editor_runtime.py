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
        self._debug_tracing = bool(self.stream_processor.config.get("debug_tracing", False))
        self._lock = threading.RLock()
        self._started = False
        self._resolution = self.stream_processor.get_resolution()
        self._reference_resolution = self.stream_processor.config.get(
            "reference_image_resolution", self._resolution
        )

        self.input_tensor = self.stream_processor.get_input_tensor()
        self.output_tensor = self.stream_processor.get_output_tensor()

        if self._debug_tracing:
            self._trace(
                "runtime initialized "
                f"input_shared_shape={tuple(self.input_tensor.shape)} "
                f"output_shared_shape={tuple(self.output_tensor.shape)} "
                f"resolution={self._resolution} "
                f"reference_resolution={self._reference_resolution}"
            )

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

    def _trace(self, message: str) -> None:
        if self._debug_tracing:
            print(f"[FluxRTDebug][editor] {message}")

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._trace(
                f"starting stream processor input_shared={self.input_tensor.name} "
                f"output_shared={self.output_tensor.name}"
            )
            self.stream_processor.start()
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._trace("stopping stream processor")
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

    def set_prompt(self, prompt: str) -> None:
        self._trace(f"set_prompt len={len(prompt)}")
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
        self._trace(
            f"set_reference_image input_shape={tuple(np.asarray(image).shape)} "
            f"cropped_shape={tuple(frame.shape)} colorspace={colorspace}"
        )
        self.stream_processor.set_reference_image(frame)

    def clear_reference_image(self) -> None:
        self._trace("clear_reference_image")
        self.stream_processor.set_reference_image(None)

    def set_canvas(self, image, *, colorspace: ColorSpace = "rgb") -> None:
        frame = self._as_numpy(image)
        frame = self._ensure_uint8(frame)
        frame = self._to_bgr(frame, colorspace)
        frame = crop_maximal_rectangle(
            frame, self._resolution["height"], self._resolution["width"]
        )
        self._trace(
            f"set_canvas input_shape={tuple(np.asarray(image).shape)} "
            f"cropped_shape={tuple(frame.shape)} colorspace={colorspace}"
        )
        self.input_tensor.copy_from(frame)

    def set_mask(self, mask, *, full_resolution: bool = True) -> None:
        mask_arr = self._as_numpy(mask)
        input_shape = tuple(mask_arr.shape)
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
            mask_arr = crop_maximal_rectangle(
                mask_arr,
                self._resolution["height"],
                self._resolution["width"],
            )
            latent_h = self._resolution["height"] // 16
            latent_w = self._resolution["width"] // 16
            mask_arr = cv2.resize(
                mask_arr,
                (latent_w, latent_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_arr = cv2.dilate(mask_arr, np.ones((3, 3), np.uint8), iterations=1)
        else:
            expected_shape = (
                self._resolution["height"] // 16,
                self._resolution["width"] // 16,
            )
            if mask_arr.shape != expected_shape:
                raise ValueError(
                    f"latent mask must have shape {expected_shape}, got {mask_arr.shape}"
                )

        active = int(np.count_nonzero(mask_arr))
        self._trace(
            "set_mask "
            f"input_shape={input_shape} processed_shape={tuple(mask_arr.shape)} "
            f"full_resolution={full_resolution} active={active}/{mask_arr.size}"
        )
        mask_arr = (mask_arr > 0).astype(np.uint8) * 2
        self.stream_processor.set_mask(mask_arr)

    def clear_mask(self) -> None:
        latent_h = self._resolution["height"] // 16
        latent_w = self._resolution["width"] // 16
        self._trace(f"clear_mask latent_shape={(latent_h, latent_w)}")
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
        self._trace(
            "update "
            f"prompt={prompt is not None} canvas={canvas is not None} mask={mask is not None} "
            f"reference_image={reference_image is not None} clear_mask={clear_mask} "
            f"clear_reference_image={clear_reference_image}"
        )
        if prompt is not None:
            self.set_prompt(prompt)
        if clear_reference_image:
            self.clear_reference_image()
        elif reference_image is not None:
            self.set_reference_image(reference_image, colorspace=reference_colorspace)
        if canvas is not None:
            self.set_canvas(canvas, colorspace=canvas_colorspace)
        if clear_mask or mask is None:
            self.clear_mask()
        else:
            self.set_mask(mask, full_resolution=full_resolution_mask)

    def get_latest_frame(self, *, colorspace: ColorSpace = "rgb") -> np.ndarray:
        with self._lock:
            frame = self.output_tensor.to_numpy()
        self._trace(f"get_latest_frame colorspace={colorspace} shape={tuple(frame.shape)}")
        return self._to_rgb(frame, "bgr") if colorspace == "rgb" else frame

    def get_shared_tensors(self):
        return self.input_tensor, self.output_tensor
