from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from stream_processor import StreamProcessor


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _build_config(args: argparse.Namespace) -> dict:
    interpolation_exp = args.interpolation_exp if args.interpolate else 0
    return {
        "model_dir": args.model_dir,
        "resolution": {"height": args.height, "width": args.width},
        "default_prompt": args.prompt,
        "default_steps": args.steps,
        "default_seed": args.seed,
        "compile_models": not args.no_compile,
        "interpolation_exp": interpolation_exp,
        "interpolate": args.interpolate,
        "rife_weights_path": args.rife_weights,
        "target_fps": args.target_fps,
        "use_reference_image": False,
        "mask_calculation_method": "auto",
        "always_update_image_cache": True,
        "logging": True,
    }


def _capture_loop(
    processor: StreamProcessor,
    camera_index: int,
    width: int,
    height: int,
    stop_event: threading.Event,
) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "opencv-python is required for webcam_stream_processor.py; install klein4B/requirements.txt"
        ) from exc

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            processor.get_input_tensor().copy_from(frame)
    finally:
        cap.release()


def serve(args: argparse.Namespace) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "opencv-python is required for webcam_stream_processor.py; install klein4B/requirements.txt"
        ) from exc

    processor = StreamProcessor(_build_config(args))
    processor.start()

    stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=_capture_loop,
        args=(processor, args.camera_index, args.width, args.height, stop_event),
        daemon=True,
    )
    capture_thread.start()

    window_name = "klein4B webcam stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    try:
        while not processor.is_ready():
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
            time.sleep(0.01)
        while True:
            frame = processor.get_output_tensor().to_numpy()
            if frame.shape[:2] != (args.height, args.width):
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                processor.set_param("prompt", args.prompt)
            if key == ord("1"):
                processor.set_steps(args.steps)
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        processor.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FluxRT-style webcam stream processor for klein4B")
    parser.add_argument(
        "--model-dir",
        default=str(_repo_root().parent / "FLUX.2-klein-4B"),
        help="Path to the FLUX.2-klein-4B model directory",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--prompt", default="a stylized artistic portrait, high quality, detailed")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--interpolation-exp", type=int, default=1)
    parser.add_argument(
        "--rife-weights",
        default=str(_repo_root().parent / "RIFE-safetensors" / "flownet.safetensors"),
    )
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile setup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve(args)


if __name__ == "__main__":
    main()
