# FLUX.2 Klein 4B

Inference pipeline for `black-forest-labs/FLUX.2-klein-4B`: T2I and I2I via 4-step distilled model

## Optimizations in place

- **torch.compile** – Transformer and VAE encode/decode are compiled. `dynamic=True` so it doesn’t recompile across resolutions.
- **Sage attention** – Default attention backend.
- **cache-dit** – Faster transformer steps via DBCache.
- **In-place Flux2 transformer patching** – `apply_flux2_transformer_klein_ops(transformer)` patches a loaded Diffusers Flux2 transformer instance to use the local CUDA-backed block helpers where the structure matches.

We tried a lighter VAE (TAEF2) for faster encode/decode; it reduced output quality, so we keep the original VAE.

## Usage

- **`klein_pipeline.py`** – Loads and run T2I/I2I.
- **`cache_dit_klein.py`** – `enable_cache_dit(pipe)` and `apply_attention_backend(pipe, "sage")` (or `"auto"` / `"native"`). Call after loading the pipeline.
- **Flux2 transformer patching** – if your notebook loads `Flux2Transformer2DModel` from Diffusers, call `apply_flux2_transformer_klein_ops(pipe.transformer)` before `pipe.enable_compile(...)`.
- **Custom CUDA denoiser hook** – Register a compiled `torch.ops` backend to replace transformer forward only:
  - `enable_cuda_denoiser_op(pipe, "klein_cuda.denoise_step", expected_hidden_tokens=..., enforce_cfg1=True)`
  - `latent_token_count_for_resolution(pipe, height=384, width=576)` helps compute expected tokens for fixed-shape runs.
- **Temporal consistency** – attach a persistent spatial cache and per-frame mask controller for webcam/video v2v:
  - `from temporal_consistency import TemporalConsistencyController, TemporalConsistencyConfig`
  - `from cache_dit_klein import enable_temporal_consistency`
  - call it after `prepare_transformer_for_speed(...)` and before `pipe.enable_compile(...)`
  - `spatial_cache = enable_temporal_consistency(pipe, height=..., width=...)`
  - `controller = TemporalConsistencyController(TemporalConsistencyConfig(height=..., width=...))`
  - `attention_kwargs = controller.build_attention_kwargs(frame_tensor, spatial_cache=spatial_cache)`
- **Webcam orchestrator** – `webcam_orchestrator.py` serves a thin local browser UI:
  - runs the `klein4B` pipeline directly
  - captures webcam frames in the browser
  - sends one frame at a time to `/infer`
  - optionally enables FluxRT-style frame interpolation for smoother playback
  - keeps temporal cache state alive across frames
  - start it with `python webcam_orchestrator.py --model-dir /path/to/FLUX.2-klein-4B`
  - add `--interpolate --interpolation-exp 1` to enable RIFE interpolation between generated frames
- **FluxRT-style webcam stream processor** – `webcam_stream_processor.py` mirrors the FluxRT split runtime:
  - model inference runs in a separate subprocess
  - output smoothing is handled by a separate scheduler subprocess
  - shared tensors carry webcam input and rendered output between processes
  - start it with `python webcam_stream_processor.py --model-dir /path/to/FLUX.2-klein-4B`
  - add `--interpolate --interpolation-exp 1` for the FluxRT-style RIFE batch smoothing
  - requires `opencv-python` from `requirements.txt`
- **Ported CUDA kernels** – `cuda_kernels/` contains specialized fused CUDA ops ported from `klein-cuda-c`.
  Build with:
  - `cd cuda_kernels && python3 setup.py build_ext --inplace`
  Then load:
  - `from cache_dit_klein import load_ported_cuda_kernels; load_ported_cuda_kernels()`
- **Option A backend bridge (strict klein-cuda-c runtime)**:
  - Build bridge:
    - `cd klein_c_bridge && bash build_bridge.sh`
  - Enable in pipeline:
    - `from cache_dit_klein import enable_klein_c_cuda_backend`
    - `enable_klein_c_cuda_backend(pipe, model_dir="/path/to/flux-klein-model")`
  - Multi-reference is supported through the same bridge path (maps token `t` offsets to klein-cuda-c reference offsets).
- **Full native path (same denoising/sampling runtime as klein-cuda-c)**:
  - `from cache_dit_klein import build_klein_c_full_backend`
  - `backend = build_klein_c_full_backend(model_dir="/path/to/flux-klein-model")`
  - `backend.img2img(prompt, image, KleinGenerateConfig(...))`
  - `backend.multiref(prompt, refs, KleinGenerateConfig(...))`
  This runs img2img/multiref through `flux_img2img` / `flux_multiref` in the C/CUDA runtime.

Enable compile after setup: `pipe.enable_compile(dynamic=True)`.

For the Colab notebook stack you shared, the intended order is:

1. Load the FP8 `Flux2Transformer2DModel`
2. Call `apply_flux2_transformer_klein_ops(transformer)`
3. Build the pipeline
4. Apply `enable_cache_dit(pipe)` and `apply_attention_backend(pipe, "auto")`
5. Call `pipe.enable_compile(...)`
