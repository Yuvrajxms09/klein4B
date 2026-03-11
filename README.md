# FLUX.2 Klein 4B — baseline inference

Minimal pipeline and inference for `black-forest-labs/FLUX.2-klein-4B`: load, T2I, I2I. No server, no compile, no warmup. Add optimizations on top as needed.

## Usage

```python
from pipeline import load_pipeline, run_t2i, run_i2i
from PIL import Image

pipe = load_pipeline()

# T2I
img = run_t2i(pipe, "A capybara under a banana leaf", seed=0)
img.save("out.png")

# I2I
ref = Image.open("out.png").convert("RGB")
img = run_i2i(pipe, "make it night time", ref, seed=1)
img.save("edit.png")
```

## Optional: lighter VAE (TAEF2)

To use the TAEF2 lightweight VAE (faster encode/decode, same latent interface), load the pipeline then swap the VAE:

```python
from klein_pipeline import Flux2KleinPipeline
from taef2_vae import replace_pipeline_vae_with_taef2
import torch

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
pipe.to("cuda")
replace_pipeline_vae_with_taef2(pipe)  # optional: cache_dir=".cache/taef2"
# then use pipe as usual for T2I / I2I
```

First run downloads `taesd.py` (from taesd repo) and `taef2.safetensors` (from [madebyollin/taef2](https://huggingface.co/madebyollin/taef2)) into `cache_dir` (default `.cache/taef2`).

## Optional: cache-DiT (faster transformer steps)

Klein 4B is a 4-step distilled model. To enable DBCache (cache conditioning across steps), call `enable_cache_dit(pipe)` after loading. The pipeline will then call `refresh_context` before each run. Defaults match **flux-stream-editor** (FastFlux2Config): `cache_fn=1`, `cache_bn=0`, `residual_diff_threshold=0.8`, `single_block_rdt_scale=3.0`, `max_warmup_steps=0`; `steps_mask` is `"10"` for 2 steps else `"1"*num_inference_steps` (so `"1111"` for 4 steps).

```python
from klein_pipeline import Flux2KleinPipeline
from cache_dit_klein import enable_cache_dit
import torch

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
pipe.to("cuda")
enable_cache_dit(pipe)  # optional: num_inference_steps=4, steps_mask=None (uses "1111" for 4 steps)
# then pipe(...) as usual; num_inference_steps=4 by default
```

Requires `pip install cache-dit`.

## Dependencies

See `requirements.txt`. Core: `torch`, `diffusers` (main), `transformers`, `accelerate`, `pillow`.

---
