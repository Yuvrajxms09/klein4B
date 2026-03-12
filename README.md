# FLUX.2 Klein 4B

Inference pipeline for `black-forest-labs/FLUX.2-klein-4B`: T2I and I2I via diffusers-style API. 4-step distilled; default `num_inference_steps=4`.

## What it does

- **`klein_pipeline.py`** – load model for T2I or I2I.
- **`taef2_vae.py`** – Optional TAEF2 lightweight VAE; faster encode/decode. Weights from [madebyollin/taef2](https://huggingface.co/madebyollin/taef2).
- **`cache_dit_klein.py`** – Optional cache-DiT and runtime opts: .

## Optimizations in place

| Component | What’s done |
|-----------|--------------|
| Pipeline | `cache_context("cond")` / `("uncond")` in denoise loop; `refresh_context` before each run when cache-dit is enabled. |
| Timesteps cache | Key `(image_seq_len, num_inference_steps, device)`; reused when same resolution/steps. Disabled when custom `sigmas` passed. |
| Image latent IDs cache | Key `(latent_h, latent_w, ...)` per image, device; reused when same resolution. |
| Prompt cache | Same prompt + encoder args → reuse `prompt_embeds` and `text_ids`. |
| Fast preprocess | Resize → numpy → tensor → normalize; `pin_memory` and `non_blocking` H2D in `prepare_image_latents`. |

Caches and fast preprocess are on by default. Toggle via `pipe._cache_timesteps`, `pipe._cache_image_latent_ids`, `pipe._cache_prompt`, `pipe._use_fast_preprocess`, `pipe._preprocess_pin_memory`, `pipe._preprocess_non_blocking_h2d`. Clear caches: `pipe.clear_inference_caches()`.

## Optional (call after load)

- **TAEF2** – `replace_pipeline_vae_with_taef2(pipe)`. Requires first-run download of `taesd.py` and `taef2.safetensors` into `.cache/taef2` (or set `cache_dir`).
