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

- **Transformer NVFP4:** all 109 transformer linear layers use TorchAO dynamic
  W4A4 NVFP4 with Triton activation quantization.
- **Qwen NVFP4:** the 189 active Qwen linear layers use the same dynamic W4A4
  NVFP4 configuration.
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
- **Latent-unpack fix:** known latent dimensions are passed directly, avoiding
  synchronizing `Tensor.item()` calls used to infer height and width.
- **Resident metadata:** timesteps, latent position IDs, RoPE tensors, and VAE
  batch-normalization constants are cached by shape, dtype, and device.
- **Fast input transfer:** preprocessing uses pinned CPU memory and non-blocking
  host-to-device copies where supported.

Prompt embeddings are cached by default. Repeating a prompt skips Qwen encoding
even if the input image changes. Encoded image contents are not cached; only
shape-dependent image position IDs are reused.

## Not enabled

The reference benchmark does not enable Cache-DiT, prediction reuse, approximate
attention, QKV fusion, custom Klein CUDA block patches, or the experimental
native backends in `cuda_kernels/`. These paths remain in the repository for
experimentation but are not part of the 113-115 ms result.

## Setup

Clone this branch and keep a Diffusers checkout beside the repository:

```bash
git clone -b optimized-nvfp4-115ms --single-branch \
  https://github.com/Yuvrajxms09/klein4B.git
git clone https://github.com/huggingface/diffusers.git
```

The Modal volume `klein4B-assets` must already contain the Klein model and input
image expected by the script. TAEF2 artifacts are fetched into the container
cache when missing. No weights are downloaded to the local machine.

Run from the `klein4B` directory:

```bash
modal run fastest_script_115ms.py
```

The script saves generated images and `benchmark.json` to
`/mnt/klein4B-assets/bench_outputs_nvfp4` after timing finishes.

## Main files

- `fastest_script_115ms.py`: Modal setup, quantization, compilation, benchmark,
  and output saving.
- `klein_pipeline.py`: optimized Klein img2img pipeline and inference caches.
- `taef2_vae.py`: TAEF2 integration.
- `cache_dit_klein.py`: optional attention, Cache-DiT, and experimental runtime
  helpers. These are not enabled by the reference benchmark.
- `cuda_kernels/`: experimental CUDA and Triton kernels, also disabled in the
  reference benchmark.
