# FLUX.2 Klein 4B

Optimized img2img inference for `black-forest-labs/FLUX.2-klein-4B` on NVIDIA
Blackwell GPUs. The reference workload is batch size 1, 576x384, four denoising
steps, guidance 1.0, and PIL output.

## Reference benchmark

[`fastest_script_115ms.py`](fastest_script_115ms.py) is the supported benchmark
for this branch. On an RTX PRO 6000 it runs at roughly 113-115 ms end to end,
excluding model loading, compilation, warmup, file writes, Modal startup, and
network latency.

The measured interval includes:

- prompt tokenization and Qwen encoding;
- input transfer and TAEF2 image encoding;
- latent creation;
- four complete transformer and scheduler steps;
- latent unpacking and normalization;
- TAEF2 decoding and conversion to an in-memory PIL image.

The source image is preprocessed once because the benchmark input already has
the target dimensions. TAEF2 encoding still runs for every measured request.

## Enabled optimizations

- **Transformer NVFP4:** transformer linear layers use TorchAO dynamic W4A4
  NVFP4 with Triton activation quantization.
- **Qwen NVFP4:** active Qwen linear layers use the same dynamic W4A4 NVFP4
  configuration.
- **Reduced Qwen execution:** Klein consumes hidden states 9, 18, and 27, so the
  text encoder stops after layer 27 and bypasses the unused language-model head.
- **TAEF2:** replaces the original VAE for faster image encoding and decoding.
- **Channels-last VAE:** TAEF2 and its decoder use channels-last memory layout.
- **Static compilation:** Qwen, the transformer, and TAEF2 decode use
  `torch.compile(mode="max-autotune", dynamic=False, fullgraph=False)`.
- **TAEF2 encode compilation:** uses `max-autotune-no-cudagraphs` because its
  output is consumed by a separately compiled graph.
- **Inductor tuning:** enables 1x1 convolution as matrix multiplication and
  coordinate-descent tuning in all directions; epilogue fusion is disabled.
- **Native attention:** exact native attention is used. Flash and Sage attention
  did not improve this NVFP4 workload.

Prompt embeddings are cached by default. Repeating a prompt skips Qwen encoding
even if the input image changes. Encoded image contents are not cached; only
shape-dependent image position IDs are reused.

## Setup

Clone this branch and keep a Diffusers checkout beside the repository:

```bash
git clone -b optimized-nvfp4-115ms --single-branch \
  https://github.com/Yuvrajxms09/klein4B.git
git clone https://github.com/huggingface/diffusers.git
```

The Modal volume `klein4B-assets` must already contain the original Klein model
and input image expected by the scripts. TAEF2 artifacts are fetched into the
container cache when missing. No weights are downloaded to the local machine.

The reference benchmark remains self-contained and performs weight quantization
when its container starts:

```bash
modal run fastest_script_115ms.py
```

An optional startup-optimized variant loads already packed weights directly
from Hugging Face. Prepare and publish its deployment artifact once:

```bash
modal run modal_prepare_nvfp4_artifact.py
```

This writes the packed transformer and 27-layer Qwen weights to
`/mnt/klein4B-assets/FLUX.2-klein-4B-torchao-nvfp4`, validates a pre-quantized
reload, and uploads the same artifact to Hugging Face. The repository is public
by default at
[`Yuvrajxms09/klein-torchao-artifacts`](https://huggingface.co/Yuvrajxms09/klein-torchao-artifacts);
pass `--private` when creating a private repository. Existing artifacts are
validated and reused unless `--force` is supplied.

Then run the separate pre-quantized benchmark. It downloads the transformer and
text encoder from the validated `Yuvrajxms09/klein-torchao-artifacts` commit;
the Modal volume is still used for the original pipeline metadata and input
image.

```bash
modal run fastest_script_prequantized_nvfp4.py
```

The script saves generated images and `benchmark.json` to
`/mnt/klein4B-assets/bench_outputs_nvfp4` after timing finishes.

## Main files

- `fastest_script_115ms.py`: supported reference benchmark with runtime weight
  quantization, compilation, timing, and output saving.
- `fastest_script_prequantized_nvfp4.py`: optional benchmark that loads the
  published NVFP4 weights directly from Hugging Face instead of quantizing them
  at container startup.
- `modal_prepare_nvfp4_artifact.py`: one-time NVFP4 weight export, reload
  validation, Modal Volume persistence, and Hugging Face upload.
- `klein_pipeline.py`: optimized Klein img2img pipeline and inference caches.
- `taef2_vae.py`: TAEF2 integration.
- `cache_dit_klein.py`: optional attention, Cache-DiT, and experimental runtime
  helpers. These are not enabled by the reference benchmark.
- `cuda_kernels/`: experimental CUDA and Triton kernels, also disabled in the
  reference benchmark.
