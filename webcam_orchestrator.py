from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageOps


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _bootstrap_import_paths() -> None:
    root = _repo_root()
    parent = root.parent
    for candidate in (root, parent / "diffusers" / "src", parent / "flux2"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _encode_jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _encode_jpeg_b64(image: Image.Image, quality: int = 90) -> str:
    return base64.b64encode(_encode_jpeg(image, quality=quality)).decode("ascii")


class WebcamSession:
    def __init__(
        self,
        *,
        model_dir: str,
        width: int,
        height: int,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        compile_model: bool,
        interpolate: bool,
        interpolation_exp: int,
        rife_weights_path: str,
    ):
        self.model_dir = model_dir
        self.width = int(width)
        self.height = int(height)
        self.prompt = prompt
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.compile_model = bool(compile_model)
        self.interpolate = bool(interpolate)
        self.interpolation_exp = int(interpolation_exp)
        self.rife_weights_path = rife_weights_path

        self.lock = threading.Lock()
        self.last_prompt: str | None = None
        self.last_frame_ms: float = 0.0
        self.previous_output_frame: Image.Image | None = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        from interpolation import expand_batch_with_rife, load_rife_ifnet
        from cache_dit_klein import (
            enable_cache_dit,
            enable_temporal_consistency,
            prepare_transformer_for_speed,
        )
        from klein_pipeline import Flux2KleinPipeline
        from temporal_consistency import TemporalConsistencyConfig, TemporalConsistencyController

        torch.set_grad_enabled(False)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        pipe = Flux2KleinPipeline.from_pretrained(
            self.model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)

        enable_cache_dit(pipe)
        prepare_transformer_for_speed(pipe, backend="auto", fuse_qkv=True)

        spatial_cache = enable_temporal_consistency(pipe, height=self.height, width=self.width)

        if self.compile_model:
            pipe.enable_compile(dynamic=True)

        self.pipe = pipe
        self.spatial_cache = spatial_cache
        self.controller = TemporalConsistencyController(
            TemporalConsistencyConfig(
                height=self.height,
                width=self.width,
                device="cuda",
                dtype=torch.bfloat16,
            )
        )
        self._expand_batch_with_rife = expand_batch_with_rife
        self._pil_to_tensor = lambda img: torch.from_numpy(
            np.asarray(img, dtype=np.uint8)
        ).permute(2, 0, 1).unsqueeze(0).to(torch.float16).div_(255.0)
        self._tensor_to_pil = lambda tensor: Image.fromarray(
            (tensor.detach().clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        )
        self.rife_model = None
        if self.interpolate and self.interpolation_exp > 0:
            self.rife_model = load_rife_ifnet(
                self.rife_weights_path,
                device="cuda",
                dtype=torch.float16,
            )
        self.pipe.clear_inference_caches()

        warmup_image = Image.new("RGB", (self.width, self.height), (128, 128, 128))
        self._infer(warmup_image, self.prompt)

    def _infer(self, frame: Image.Image, prompt: str) -> list[Image.Image]:
        frame = frame.convert("RGB")
        frame = ImageOps.fit(frame, (self.width, self.height), method=Image.Resampling.BILINEAR)

        if prompt != self.last_prompt:
            self.pipe.clear_inference_caches()
            self.last_prompt = prompt

        attention_kwargs = self.controller.build_attention_kwargs(
            frame,
            spatial_cache=self.spatial_cache,
        )

        generator = torch.Generator(device="cuda").manual_seed(self.seed)
        t0 = time.perf_counter()
        output = self.pipe(
            prompt=prompt,
            image=frame,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            output_type="pil",
            attention_kwargs=attention_kwargs,
        )
        self.last_frame_ms = (time.perf_counter() - t0) * 1000.0
        current_frame = output.images[0].convert("RGB")

        if self.previous_output_frame is None:
            self.previous_output_frame = current_frame
        
        if not self.interpolate or self.rife_model is None or self.interpolation_exp < 1:
            self.previous_output_frame = current_frame
            return [current_frame]

        previous_tensor = self._pil_to_tensor(self.previous_output_frame).to(
            device="cuda", dtype=torch.float16
        )
        current_tensor = self._pil_to_tensor(current_frame).to(
            device="cuda", dtype=torch.float16
        )
        batch = self._expand_batch_with_rife(
            previous_tensor,
            current_tensor,
            self.rife_model,
            self.interpolation_exp,
        )
        frames = [self._tensor_to_pil(batch[i]) for i in range(batch.shape[0])]
        self.previous_output_frame = current_frame
        return frames

    def process_request(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> dict[str, object]:
        with self.lock:
            self.num_inference_steps = int(num_inference_steps)
            self.guidance_scale = float(guidance_scale)
            self.seed = int(seed)
            decode_t0 = time.perf_counter()
            frame = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            decoded_ms = (time.perf_counter() - decode_t0) * 1000.0

            infer_t0 = time.perf_counter()
            results = self._infer(frame, prompt)
            infer_ms = (time.perf_counter() - infer_t0) * 1000.0

            frame_count = len(results)
            display_interval_ms = self.last_frame_ms / frame_count if frame_count > 0 else self.last_frame_ms
            meta = {
                "status": "ok",
                "decode_ms": round(decoded_ms, 3),
                "infer_ms": round(infer_ms, 3),
                "frame_ms": round(self.last_frame_ms, 3),
                "frame_count": frame_count,
                "display_interval_ms": round(display_interval_ms, 3),
                "interpolate": self.interpolate and self.rife_model is not None and self.interpolation_exp > 0,
                "width": self.width,
                "height": self.height,
            }
            return {
                "status": "ok",
                "frames": [_encode_jpeg_b64(result) for result in results],
                "meta": meta,
            }


class WebcamHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "klein4B-webcam/1.0"

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self.server.html.encode("utf-8"))  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/infer":
            self.send_error(404)
            return

        query = urllib.parse.parse_qs(parsed.query)
        prompt = query.get("prompt", [self.server.session.prompt])[0]  # type: ignore[attr-defined]
        steps = int(query.get("steps", [self.server.session.num_inference_steps])[0])  # type: ignore[attr-defined]
        guidance = float(query.get("guidance", [self.server.session.guidance_scale])[0])  # type: ignore[attr-defined]
        seed = int(query.get("seed", [self.server.session.seed])[0])  # type: ignore[attr-defined]

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            payload = self.server.session.process_request(
                body,
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed,
            )  # type: ignore[attr-defined]
        except Exception as exc:
            self._json_response(500, {"status": "error", "error": repr(exc)})
            return

        self.send_response(200)
        body = json.dumps(payload).encode("utf-8")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return


def _render_html(default_prompt: str, width: int, height: int, interpolate_enabled: bool) -> str:
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>klein4B webcam</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #07080b;
        --panel: rgba(255,255,255,.05);
        --line: rgba(255,255,255,.10);
        --text: #f5f7fb;
        --muted: rgba(245,247,251,.70);
        --accent: #8fe3ff;
      }}
      html, body {{ margin: 0; height: 100%; background: radial-gradient(circle at top, #121826 0%, var(--bg) 55%); color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
      .wrap {{ min-height: 100%; padding: 24px; box-sizing: border-box; display: grid; place-items: center; }}
      .shell {{ width: min(1240px, 100%); display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }}
      .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 16px; backdrop-filter: blur(14px); }}
      .title {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; gap: 16px; }}
      .title h1 {{ margin: 0; font-size: 18px; letter-spacing: .02em; }}
      .title span {{ color: var(--muted); font-size: 12px; }}
      video, img, canvas {{ width: 100%; aspect-ratio: {width} / {height}; background: #0d1117; border-radius: 16px; object-fit: cover; }}
      textarea, input, button {{ box-sizing: border-box; width: 100%; }}
      textarea, input {{ border: 1px solid var(--line); border-radius: 14px; background: rgba(0,0,0,.28); color: var(--text); padding: 12px; }}
      textarea {{ min-height: 108px; resize: vertical; }}
      button {{ margin-top: 10px; border: 0; border-radius: 14px; padding: 12px 14px; font-weight: 700; background: linear-gradient(135deg, #ffffff, #cdefff); color: #07111a; cursor: pointer; }}
      .grid {{ display: grid; gap: 12px; }}
      .meta {{ display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 12px; margin-top: 10px; }}
      .row {{ display: grid; gap: 12px; grid-template-columns: 1fr 120px 120px; }}
      @media (max-width: 960px) {{ .shell {{ grid-template-columns: 1fr; }} .row {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="shell">
        <div class="panel">
          <div class="title">
            <h1>Camera</h1>
            <span id="camera-status">idle</span>
          </div>
          <video id="video" autoplay playsinline muted></video>
          <div class="meta">
            <span>browser webcam input</span>
            <span>{width} x {height}</span>
          </div>
        </div>
        <div class="panel">
          <div class="title">
            <h1>Output</h1>
            <span id="infer-status">waiting</span>
          </div>
      <img id="output" alt="styled output" />
      <div class="meta">
        <span id="latency">0 ms</span>
        <span id="fps">0 fps</span>
      </div>
    </div>
    <div class="panel" style="grid-column: 1 / -1;">
      <div class="grid">
        <textarea id="prompt">{default_prompt}</textarea>
        <div class="row">
          <input id="steps" type="number" min="1" max="8" value="4" />
          <input id="guidance" type="number" min="0" step="0.1" value="1.0" />
          <input id="seed" type="number" value="42" />
        </div>
        <label style="display:flex;align-items:center;gap:10px;color:var(--muted);font-size:13px;">
          <input id="interpolate" type="checkbox" style="width:auto;" />
          Enable frame interpolation
        </label>
        <button id="start">Start webcam</button>
      </div>
    </div>
      </div>
    </div>
    <script>
      const video = document.getElementById("video");
      const output = document.getElementById("output");
      const promptEl = document.getElementById("prompt");
      const stepsEl = document.getElementById("steps");
      const guidanceEl = document.getElementById("guidance");
      const seedEl = document.getElementById("seed");
      const interpolateEl = document.getElementById("interpolate");
      const startBtn = document.getElementById("start");
      const cameraStatus = document.getElementById("camera-status");
      const inferStatus = document.getElementById("infer-status");
      const latencyEl = document.getElementById("latency");
      const fpsEl = document.getElementById("fps");
      const capture = document.createElement("canvas");
      capture.width = {width};
      capture.height = {height};
      const ctx = capture.getContext("2d", {{ willReadFrequently: false }});
      let inFlight = false;
      let started = false;
      let frames = 0;
      let fpsWindowStart = performance.now();
      let lastInfer = 0;
      let renderToken = 0;
      interpolateEl.checked = {str(bool(interpolate_enabled)).lower()};
      interpolateEl.disabled = true;

      function frameUrl(base64) {{
        return `data:image/jpeg;base64,${{base64}}`;
      }}

      async function playFrames(frameBatch, displayIntervalMs, token) {{
        for (let i = 0; i < frameBatch.length; i += 1) {{
          if (token !== renderToken) return;
          output.src = frameUrl(frameBatch[i]);
          if (i < frameBatch.length - 1) {{
            await new Promise((resolve) => setTimeout(resolve, displayIntervalMs));
          }}
        }}
      }}

      async function sendFrame() {{
        if (!started || inFlight || video.readyState < 2) return;
        inFlight = true;
        ctx.drawImage(video, 0, 0, capture.width, capture.height);
        capture.toBlob(async (blob) => {{
          if (!blob) {{
            inFlight = false;
            return;
          }}
          const params = new URLSearchParams({{
            prompt: promptEl.value,
            steps: stepsEl.value,
            guidance: guidanceEl.value,
            seed: seedEl.value,
          }});
          const t0 = performance.now();
          const token = ++renderToken;
          try {{
            const response = await fetch(`/infer?${{params.toString()}}`, {{
              method: "POST",
              body: blob,
            }});
            if (!response.ok) {{
              inferStatus.textContent = "error";
              inFlight = false;
              return;
            }}
            const payload = await response.json();
            const meta = payload.meta || {{}};
            const frameBatch = Array.isArray(payload.frames) ? payload.frames : [];
            const interval = Math.max(16, Math.round(meta.display_interval_ms || 16));
            await playFrames(frameBatch, interval, token);
            lastInfer = performance.now() - t0;
            latencyEl.textContent = `${{Math.round(lastInfer)}} ms`;
            inferStatus.textContent = "ok";
            if (meta.interpolate) {{
              cameraStatus.textContent = `live · interp x${{frameBatch.length}}`;
            }} else {{
              cameraStatus.textContent = "live";
            }}
            frames += 1;
            const now = performance.now();
            if (now - fpsWindowStart > 1000) {{
              fpsEl.textContent = `${{Math.round((frames * 1000) / (now - fpsWindowStart))}} fps`;
              fpsWindowStart = now;
              frames = 0;
            }}
          }} catch (err) {{
            inferStatus.textContent = "offline";
          }} finally {{
            inFlight = false;
          }}
        }}, "image/jpeg", 0.92);
      }}

      async function start() {{
        const stream = await navigator.mediaDevices.getUserMedia({{
          video: {{ width: {width}, height: {height}, facingMode: "user" }},
          audio: false,
        }});
        video.srcObject = stream;
        started = true;
        cameraStatus.textContent = "live";
        inferStatus.textContent = "ready";
        if (interpolateEl.checked) {{
          cameraStatus.textContent = "live · interpolation on";
        }}
        setInterval(sendFrame, 60);
      }}

      startBtn.onclick = () => {{
        startBtn.disabled = true;
        start().catch((err) => {{
          cameraStatus.textContent = "camera denied";
          startBtn.disabled = false;
          console.error(err);
        }});
      }};
    </script>
  </body>
</html>
"""


def serve(
    *,
    model_dir: str,
    host: str,
    port: int,
    width: int,
    height: int,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    compile_model: bool,
    interpolate: bool,
    interpolation_exp: int,
    rife_weights_path: str,
) -> None:
    session = WebcamSession(
        model_dir=model_dir,
        width=width,
        height=height,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        compile_model=compile_model,
        interpolate=interpolate,
        interpolation_exp=interpolation_exp,
        rife_weights_path=rife_weights_path,
    )

    html = _render_html(prompt, width, height, interpolate)

    class _Server(ThreadingHTTPServer):
        pass

    server = _Server((host, port), WebcamHTTPRequestHandler)
    server.session = session  # type: ignore[attr-defined]
    server.html = html  # type: ignore[attr-defined]

    print(f"webcam server listening on http://{host}:{port}")
    print(f"model_dir={model_dir}")
    print(f"resolution={width}x{height} steps={num_inference_steps} guidance={guidance_scale} seed={seed}")
    print(f"interpolate={interpolate} interpolation_exp={interpolation_exp} rife_weights={rife_weights_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin webcam orchestrator for klein4B")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("KLEIN4B_MODEL_DIR", str(_repo_root().parent / "FLUX.2-klein-4B")),
        help="Path to the FLUX.2-klein-4B model directory",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--prompt", default="a stylized artistic portrait, high quality, detailed")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile setup")
    parser.add_argument("--interpolate", action="store_true", help="Enable FluxRT-style RIFE interpolation between generated frames")
    parser.add_argument("--interpolation-exp", type=int, default=1, help="RIFE interpolation exponent; 1 yields one midpoint frame")
    parser.add_argument(
        "--rife-weights",
        default=os.environ.get("KLEIN4B_RIFE_WEIGHTS", str(_repo_root().parent / "RIFE-safetensors" / "flownet.safetensors")),
        help="Path to RIFE flownet.safetensors",
    )
    return parser.parse_args()


def main() -> None:
    _bootstrap_import_paths()
    args = parse_args()
    serve(
        model_dir=args.model_dir,
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        compile_model=not args.no_compile,
        interpolate=args.interpolate,
        interpolation_exp=args.interpolation_exp,
        rife_weights_path=args.rife_weights,
    )


if __name__ == "__main__":
    main()
