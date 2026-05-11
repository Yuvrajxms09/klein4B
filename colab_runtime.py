from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Literal
import time

import numpy as np

from editor_loop import EditorState, FluxRTEditorLoop
from editor_runtime import FluxRTEditorRuntime


ColorSpace = Literal["rgb", "bgr"]


@dataclass
class ColabRuntimeSession:
    runtime: FluxRTEditorRuntime
    loop: FluxRTEditorLoop

    def close(self) -> None:
        self.runtime.stop()

    def latest_frame(self, *, colorspace: ColorSpace = "rgb") -> np.ndarray:
        return self.loop.latest_frame(colorspace=colorspace)

    def wait_until_ready(self, timeout_s: float | None = None) -> bool:
        return self.loop.wait_until_ready(timeout_s=timeout_s)


def _resolve_default_config(config_path: str | None) -> str:
    if config_path is None:
        config_path = "fluxrt_configs/config_with_reference.json"

    path = Path(config_path)
    if path.is_absolute():
        return str(path)

    repo_root = Path(__file__).resolve().parent
    candidate = (repo_root / path).resolve()
    if candidate.exists():
        return str(candidate)
    return str(path.resolve())


def _frame_hash(frame: np.ndarray) -> str:
    arr = np.ascontiguousarray(frame)
    hasher = blake2b(digest_size=16)
    hasher.update(arr.shape.__repr__().encode("utf-8"))
    hasher.update(arr.dtype.str.encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _to_display_object(frame: np.ndarray):
    from PIL import Image

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    return Image.fromarray(frame)


def start_colab_runtime(
    config_path: str | None = None,
    *,
    wait_until_ready: bool = True,
    ready_timeout_s: float | None = 300.0,
) -> ColabRuntimeSession:
    runtime = FluxRTEditorRuntime(_resolve_default_config(config_path), start=True)
    session = ColabRuntimeSession(runtime=runtime, loop=FluxRTEditorLoop(runtime))
    if wait_until_ready:
        session.wait_until_ready(timeout_s=ready_timeout_s)
    return session


def update_editor_state(
    session: ColabRuntimeSession,
    *,
    prompt: str | None = None,
    canvas=None,
    mask=None,
    reference_image=None,
    canvas_colorspace: ColorSpace = "rgb",
    reference_colorspace: ColorSpace = "rgb",
    full_resolution_mask: bool = True,
) -> bool:
    return session.loop.step(
        EditorState(
            prompt=prompt,
            canvas=canvas,
            mask=mask,
            reference_image=reference_image,
            canvas_colorspace=canvas_colorspace,
            reference_colorspace=reference_colorspace,
            full_resolution_mask=full_resolution_mask,
        )
    )


def show_latest_frame(
    session: ColabRuntimeSession,
    *,
    colorspace: ColorSpace = "rgb",
):
    frame = session.latest_frame(colorspace=colorspace)
    try:
        from IPython.display import display
    except Exception:
        return frame
    display(_to_display_object(frame))
    return frame


def display_live_preview(
    session: ColabRuntimeSession,
    *,
    seconds: float | None = 10.0,
    interval_s: float = 0.1,
    colorspace: ColorSpace = "rgb",
    title: str | None = None,
):
    try:
        from IPython.display import clear_output, display
    except Exception:
        last_frame = None
        try:
            if seconds is None:
                while True:
                    last_frame = session.latest_frame(colorspace=colorspace)
                    time.sleep(interval_s)
            else:
                deadline = time.time() + max(seconds, 0.0)
                while time.time() < deadline:
                    last_frame = session.latest_frame(colorspace=colorspace)
                    time.sleep(interval_s)
        except KeyboardInterrupt:
            pass
        return last_frame

    last_hash = None
    last_frame = None

    try:
        if seconds is None:
            while True:
                frame = session.latest_frame(colorspace=colorspace)
                last_frame = frame
                frame_hash = _frame_hash(frame)

                if frame_hash != last_hash:
                    clear_output(wait=True)
                    if title is not None:
                        print(title)
                    display(_to_display_object(frame))
                    last_hash = frame_hash

                time.sleep(max(interval_s, 0.01))
        else:
            deadline = time.time() + max(seconds, 0.0)
            while time.time() < deadline:
                frame = session.latest_frame(colorspace=colorspace)
                last_frame = frame
                frame_hash = _frame_hash(frame)

                if frame_hash != last_hash:
                    clear_output(wait=True)
                    if title is not None:
                        print(title)
                    display(_to_display_object(frame))
                    last_hash = frame_hash

                time.sleep(max(interval_s, 0.01))
    except KeyboardInterrupt:
        pass

    return last_frame
