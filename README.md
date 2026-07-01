# FLUX.2 Klein 4B

Inference pipeline for `black-forest-labs/FLUX.2-klein-4B`: T2I and I2I via 4-step distilled model

## Optimizations in place

- **torch.compile** – Transformer and VAE encode/decode are compiled. `dynamic=True` so it doesn’t recompile across resolutions.
- **Sage attention** – Default attention backend.
- **FluxRT stream consistency** – Shared-memory stream processing, RIFE interpolation, changed-region masking, and spatial cache reuse for webcam and v2v workflows.
- **Temporal consistency** – Repeated prompt calls reuse the previous frame via partial denoising for smoother iterative updates; `Flux2KleinPipeline.refine_frame(...)` exposes the same path explicitly.

We tried a lighter VAE (TAEF2) for faster encode/decode; it reduced output quality, so we keep the original VAE.

## Usage

- **`klein_pipeline.py`** – Loads and run T2I/I2I.
- **Temporal loop** – Repeated `pipe(...)` calls with the same prompt will now refine the previous output by default. Use `feedback_strength=0` to disable it or `clear_temporal_state()` to reset it.
- **Direct editor bridge** – [`editor_runtime.py`](/Users/yuvraj/Desktop/multi-angle/klein4B/editor_runtime.py) exposes `FluxRTEditorRuntime` for non-Gradio editor integration: push canvas frames, prompt, and masks directly into `StreamProcessor`, then poll the latest output frame.
- **Editor loop helper** – [`editor_loop.py`](/Users/yuvraj/Desktop/multi-angle/klein4B/editor_loop.py) adds `FluxRTEditorLoop.step(...)` for app render loops that want to dedupe repeated canvas/mask/prompt updates before forwarding them to FluxRT.
- **Colab helper** – [`colab_runtime.py`](/Users/yuvraj/Desktop/multi-angle/klein4B/colab_runtime.py) adds `start_colab_runtime(...)`, `update_editor_state(...)`, `show_latest_frame(...)`, and `display_live_preview(...)` for persistent notebook sessions.

Enable compile after setup: `pipe.enable_compile(dynamic=True)`.
