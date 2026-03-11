# FLUX.2 Klein 4B

Inference pipeline for `black-forest-labs/FLUX.2-klein-4B`: T2I and I2I via diffusers-style API. 4-step distilled; default `num_inference_steps=4`.

## What it does

- **`klein_pipeline.py`** – `Flux2KleinPipeline`: load model, `pipe(prompt=..., image=..., height=..., width=...)` for T2I or I2I.
- **`taef2_vae.py`** – Optional TAEF2 lightweight VAE: `replace_pipeline_vae_with_taef2(pipe)` after load. Same latent API; faster encode/decode. Weights from [madebyollin/taef2](https://huggingface.co/madebyollin/taef2).
- **`cache_dit_klein.py`** – Optional cache-DiT and runtime opts: `enable_cache_dit(pipe)`, `apply_attention_backend(pipe, ...)`, `apply_transformer_compile(pipe)`.

## Optimizations in place

| Component | What’s done |
|-----------|--------------|
| Pipeline | `cache_context("cond")` / `("uncond")` in denoise loop; `refresh_context` before each run when cache-dit is enabled. |
| Denoise loop | `torch.compiler.cudagraph_mark_step_begin()` at each step when available. |
| Defaults | 4 steps, no progress bar change (call `pipe.set_progress_bar_config(disable=True)` in notebook if desired). |

## Optional (call after load)

- **TAEF2** – `replace_pipeline_vae_with_taef2(pipe)`. Requires first-run download of `taesd.py` and `taef2.safetensors` into `.cache/taef2` (or set `cache_dir`).
- **Cache-DiT** – `enable_cache_dit(pipe)`. Requires `pip install cache-dit`. Defaults match flux-stream-editor (4-step: steps_mask `"1111"`).
- **Attention backend** – `apply_attention_backend(pipe, "sage" \| "native" \| "fa3" \| "auto")`. Must be called; installing flash/sage alone is not enough.
- **Transformer compile** – `apply_transformer_compile(pipe)`. First run after this is slow (warmup). With variable resolutions you may want to skip or warm fixed resolutions.
- **Notebook-only** – `pipe.transformer.fuse_qkv_projections()`, `pipe.vae.fuse_qkv_projections()`, `pipe.vae.to(memory_format=torch.channels_last)`; TF32: `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`.

## Dependencies

See `requirements.txt`. Core: `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`, `pillow`. Optional: `cache-dit` for cache-DiT.
