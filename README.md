# FLUX.2 Klein 4B

Inference pipeline for `black-forest-labs/FLUX.2-klein-4B`: T2I and I2I via 4-step distilled model

## Optimizations in place

- **torch.compile** – Transformer and VAE encode/decode are compiled. `dynamic=True` so it doesn’t recompile across resolutions.
- **Sage attention** – Default attention backend.
- **cache-dit** – Faster transformer steps via DBCache.
- **Temporal consistency** – Repeated prompt calls reuse the previous frame via partial denoising for smoother iterative updates; `Flux2KleinPipeline.refine_frame(...)` exposes the same path explicitly.

We tried a lighter VAE (TAEF2) for faster encode/decode; it reduced output quality, so we keep the original VAE.

## Usage

- **`klein_pipeline.py`** – Loads and run T2I/I2I.
- **`cache_dit_klein.py`** – `enable_cache_dit(pipe)` and `apply_attention_backend(pipe, "sage")` (or `"auto"` / `"native"`). Call after loading the pipeline.
- **Temporal loop** – Repeated `pipe(...)` calls with the same prompt will now refine the previous output by default. Use `feedback_strength=0` to disable it or `clear_temporal_state()` to reset it.

Enable compile after setup: `pipe.enable_compile(dynamic=True)`.
