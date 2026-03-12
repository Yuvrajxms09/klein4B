# FLUX.2 Klein 4B

Inference pipeline for `black-forest-labs/FLUX.2-klein-4B`: T2I and I2I via diffusers-style API. 4-step distilled; default `num_inference_steps=4`.

## Optimizations in place

- **torch.compile** – Transformer and VAE encode/decode are compiled. `dynamic=True` so it doesn’t recompile across resolutions.
- **Sage attention** – Default attention backend.
- **cache-dit** – Faster transformer steps via DBCache.

We tried a lighter VAE (TAEF2) for faster encode/decode; it reduced output quality, so we keep the original VAE.

## Usage

- **`klein_pipeline.py`** – Loads and run T2I/I2I.
- **`cache_dit_klein.py`** – `enable_cache_dit(pipe)` and `apply_attention_backend(pipe, "sage")` (or `"auto"` / `"native"`). Call after loading the pipeline.

Enable compile after setup: `pipe.enable_compile(dynamic=True)`.
