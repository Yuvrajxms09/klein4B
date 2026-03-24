# FLUX.2 Klein 4B

Inference pipeline for `black-forest-labs/FLUX.2-klein-4B`: T2I and I2I via 4-step distilled model

## Optimizations in place

- **torch.compile** – Transformer and VAE encode/decode are compiled. `dynamic=True` so it doesn’t recompile across resolutions.
- **Sage attention** – Default attention backend.
- **cache-dit** – Faster transformer steps via DBCache.
- **Optional FP8 experimental path** – `pipe.enable_fp8_optimizations(...)` can replace linear layers with an FP8 `torch._scaled_mm` path using calibrated scales auto-loaded from checkpoint/HF.
- **Optional RoPE no-upcast patch** – enabled via `pipe.enable_fp8_optimizations(patch_rope=True)` to avoid per-call FP32 upcasting in Flux2 rotary embedding.

We tried a lighter VAE (TAEF2) for faster encode/decode; it reduced output quality, so we keep the original VAE.

## Usage

- **`klein_pipeline.py`** – Loads and run T2I/I2I.
- **`cache_dit_klein.py`** – `enable_cache_dit(pipe)` and `apply_attention_backend(pipe, "sage")` (or `"auto"` / `"native"`). Call after loading the pipeline.

Enable compile after setup: `pipe.enable_compile(dynamic=True)`.

## Experimental FP8 usage

Use only when your checkpoint exposes calibrated scales.

```python
from klein_pipeline import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
summary = pipe.enable_fp8_optimizations(
    scales_checkpoint_path="/path/to/model-or-transformer-dir",
    # OR: hf_repo_id="black-forest-labs/FLUX.2-klein-base-9B",
    quantize_backend="triton",   # "compile" or "triton"
    require_full_coverage=True,  # fail fast if any linear is not FP8-mapped
    patch_rope=True,
)
print(summary)
```

### TorchAO static-FP8 checkpoint loader (Photoroom format)

```python
from diffusers import Flux2Transformer2DModel
from huggingface_hub import hf_hub_download
from fp8_optimizations import load_torchao_fp8_static_model

ckpt_path = hf_hub_download(
    "photoroom/FLUX.2-klein-4b-fp8-diffusers",
    filename="transformer_fp8_static/model_fp8_static.pt",
)

transformer, load_report = load_torchao_fp8_static_model(
    ckpt_path=ckpt_path,
    base_model_or_factory=lambda: Flux2Transformer2DModel.from_pretrained(
        "photoroom/FLUX.2-klein-4b-fp8-diffusers",
        subfolder="transformer_bf16",
        torch_dtype=torch.bfloat16,
    ),
    device="cuda",
    quantize_backend="compile",
    require_full_coverage=True,
    verbose_print=True,  # print per-load diagnostics in notebook logs
)
print(load_report)
```
