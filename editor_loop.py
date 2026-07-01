from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Literal

import numpy as np

from editor_runtime import FluxRTEditorRuntime


ColorSpace = Literal["rgb", "bgr"]


@dataclass(frozen=True)
class EditorState:
    prompt: str | None = None
    canvas: np.ndarray | None = None
    mask: np.ndarray | None = None
    reference_image: np.ndarray | None = None
    canvas_colorspace: ColorSpace = "rgb"
    reference_colorspace: ColorSpace = "rgb"
    full_resolution_mask: bool = True
    clear_mask: bool = False
    clear_reference_image: bool = False


class FluxRTEditorLoop:
    """
    Small app-loop helper for embedding FluxRTEditorRuntime in a real editor UI.

    The UI should call `step(...)` whenever the current editor state changes.
    The helper deduplicates state and forwards only changed fields to FluxRT.
    """

    def __init__(self, runtime: FluxRTEditorRuntime):
        self.runtime = runtime
        self._last_prompt: str | None = None
        self._last_canvas_key: tuple[str | None, ColorSpace] | None = None
        self._last_mask_key: tuple[str | None, bool] | None = None
        self._last_reference_key: tuple[str | None, ColorSpace] | None = None

    @staticmethod
    def _hash_array(array: np.ndarray | None) -> str | None:
        if array is None:
            return None
        arr = np.ascontiguousarray(array)
        hasher = blake2b(digest_size=16)
        hasher.update(arr.shape.__repr__().encode("utf-8"))
        hasher.update(arr.dtype.str.encode("utf-8"))
        hasher.update(arr.tobytes())
        return hasher.hexdigest()

    def step(self, state: EditorState) -> bool:
        changed = False

        if state.prompt is not None and state.prompt != self._last_prompt:
            self.runtime.set_prompt(state.prompt)
            self._last_prompt = state.prompt
            changed = True

        canvas_hash = self._hash_array(state.canvas)
        canvas_key = (canvas_hash, state.canvas_colorspace)
        if state.canvas is not None and canvas_key != self._last_canvas_key:
            self.runtime.set_canvas(state.canvas, colorspace=state.canvas_colorspace)
            self._last_canvas_key = canvas_key
            changed = True

        mask_hash = self._hash_array(state.mask)
        mask_key = (mask_hash, state.full_resolution_mask)
        if state.mask is not None and mask_key != self._last_mask_key:
            self.runtime.set_mask(
                state.mask,
                full_resolution=state.full_resolution_mask,
            )
            self._last_mask_key = mask_key
            changed = True
        elif state.clear_mask or self._last_mask_key is not None:
            self.runtime.clear_mask()
            self._last_mask_key = None
            changed = True

        reference_hash = self._hash_array(state.reference_image)
        reference_key = (reference_hash, state.reference_colorspace)
        if state.clear_reference_image:
            self.runtime.clear_reference_image()
            self._last_reference_key = None
            changed = True
        elif (
            state.reference_image is not None
            and reference_key != self._last_reference_key
        ):
            self.runtime.set_reference_image(
                state.reference_image,
                colorspace=state.reference_colorspace,
            )
            self._last_reference_key = reference_key
            changed = True

        return changed

    def latest_frame(self, *, colorspace: ColorSpace = "rgb") -> np.ndarray:
        return self.runtime.get_latest_frame(colorspace=colorspace)

    def wait_until_ready(self, timeout_s: float | None = None) -> bool:
        return self.runtime.wait_until_ready(timeout_s=timeout_s)
