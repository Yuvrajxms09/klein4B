# FLUX.2 Klein 4B NVFP4 Inference

Optimized image-to-image inference for
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
on NVIDIA Blackwell GPUs. The validated workload is batch size 1, 576x384,
four denoising steps, guidance 1.0, and PIL output.

The reference result on an RTX PRO 6000 is approximately **113-115 ms** for a
new prompt. Repeating a cached prompt reduces the measured request to
approximately **102 ms** by skipping Qwen prompt encoding.

## Entry Points

| File | Weight source | Startup behavior | Intended use |
| --- | --- | --- | --- |
| [`fastest_script_115ms.py`](fastest_script_115ms.py) | BF16 checkpoint on the Modal volume | Quantizes transformer and Qwen weights to NVFP4 when the container starts | Self-contained reference benchmark |
| [`fastest_script_prequantized_nvfp4.py`](fastest_script_prequantized_nvfp4.py) | [Pre-quantized Hugging Face artifact](https://huggingface.co/Yuvrajxms09/klein-torchao-artifacts) | Loads packed NVFP4 weights without repeating weight quantization | Deployment-oriented benchmark and cold-start comparison |
| [`modal_prepare_nvfp4_artifact.py`](modal_prepare_nvfp4_artifact.py) | BF16 checkpoint on the Modal volume | Builds, validates, saves, and publishes the NVFP4 artifact | One-time artifact generation |

Both inference scripts execute the same W4A4 path. Weights are NVFP4, and
request-dependent BF16 activations are dynamically quantized to NVFP4 before
each quantized linear operation. Saving the weights removes startup weight
quantization; it does not remove runtime activation quantization or change
steady-state inference latency.

## Validated Artifacts

- Source model:
  [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- NVFP4 artifact:
  [`Yuvrajxms09/klein-torchao-artifacts`](https://huggingface.co/Yuvrajxms09/klein-torchao-artifacts)
- Validated artifact revision: `7fd5321202831de6c833bd55f318c2f5e92a1bdd`
- Repository branch:
  [`optimized-nvfp4-115ms`](https://github.com/Yuvrajxms09/klein4B/tree/optimized-nvfp4-115ms)

The artifact contains:

- 109 pre-quantized transformer linear layers;
- 189 pre-quantized Qwen linear layers across the first 27 decoder layers;
- dynamic NVFP4 activation-quantization configuration;
- a manifest recording the serialization formats and package versions.

The transformer is stored using Diffusers' supported PyTorch serialization
path. The Qwen encoder uses Transformers' TorchAO safetensors serialization.
Both components are validated by a clean pre-quantized reload before the
artifact is committed or uploaded.

## Inference Configuration

- **Transformer:** TorchAO dynamic W4A4 NVFP4.
- **Text encoder:** Qwen3 reduced from 36 to 27 decoder layers and quantized to
  dynamic W4A4 NVFP4. Klein consumes hidden states 9, 18, and 27, so later
  layers and the language-model head do not contribute to prompt embeddings.
- **VAE:** TAEF2 replaces the original VAE for image encoding and decoding.
- **Memory layout:** TAEF2 and its decoder use channels-last layout.
- **Attention:** exact native attention.
- **Compilation:** Qwen, transformer, and TAEF2 decode use
  `torch.compile(mode="max-autotune", dynamic=False, fullgraph=False)`.
- **TAEF2 encode:** uses `max-autotune-no-cudagraphs` because its output crosses
  into a separately compiled transformer graph.
- **Inductor:** enables 1x1 convolution as matrix multiplication and
  coordinate-descent tuning in all directions; epilogue fusion is disabled.
- **Prompt cache:** repeated prompts reuse prompt embeddings even when the input
  image changes.

## Timing Contract

The measured request includes:

- prompt tokenization and Qwen encoding for uncached prompts;
- input transfer and TAEF2 image encoding;
- latent and noise creation;
- four complete transformer and scheduler steps;
- latent unpacking and normalization;
- TAEF2 decoding and conversion to an in-memory PIL image.

The measured request excludes:

- Modal startup and image construction;
- model download and loading;
- weight quantization during container initialization;
- `torch.compile` compilation and autotuning;
- warmup requests;
- deterministic CPU image preprocessing;
- writing generated images and JSON results to disk;
- server and network latency.

TAEF2 image encoding is still executed for every request. Only CPU
preprocessing is moved outside timing because the benchmark image already has
the target dimensions.

## Observed Results

The validated pre-quantized run used 20 unique prompts followed by four repeated
prompts:

| Request type | Mean | Median | p95 |
| --- | ---: | ---: | ---: |
| New prompt | 114.07 ms | 113.79 ms | 115.39 ms |
| Cached prompt | 102.15 ms | 102.31 ms | 102.50 ms |

Do not use the combined 24-request mean as the fresh-prompt E2E result because
it mixes cached and uncached requests. Loading pre-quantized weights improves
initialization, not these steady-state request times.

## Requirements

- NVIDIA Blackwell GPU with native NVFP4 support. The validated Modal GPU is
  `RTX-PRO-6000`.
- Modal volume named `klein4B-assets`.
- Modal secret named `huggingface-secret` with `HF_TOKEN` when publishing or
  accessing gated artifacts.
- Original model at
  `/mnt/klein4B-assets/FLUX.2-klein-4B`.
- Input image at `/mnt/klein4B-assets/calib/blue_car.jpeg`.
- Local Diffusers checkout beside the `klein4B` repository.

The Modal images pin the runtime used by the pre-quantized artifact:

- PyTorch `2.11.0` with CUDA 13.0 wheels;
- TorchAO `0.17.0`;
- MSLK CUDA `1.1.0`;
- Transformers revision `63f32a8782cb70da3365acab16f2b67947737985`.

## Setup

Clone the optimized branch and Diffusers beside it:

```bash
git clone -b optimized-nvfp4-115ms --single-branch \
  https://github.com/Yuvrajxms09/klein4B.git
git clone https://github.com/huggingface/diffusers.git
cd klein4B
```

Populate the Modal volume with the original checkpoint and input image using
[`upload.py`](upload.py), or provide the same paths by another deployment
process.

## Run The Reference Benchmark

This path loads BF16 weights from the Modal volume and quantizes them when the
container starts:

```bash
modal run fastest_script_115ms.py
```

## Run With Pre-Quantized Weights

This path downloads the transformer and text encoder directly from the pinned
Hugging Face artifact. The original model directory remains necessary for the
tokenizer, scheduler, and other pipeline components.

```bash
modal run fastest_script_prequantized_nvfp4.py
```

On load, the script verifies:

- both Hugging Face loaders selected their pre-quantized paths;
- the transformer contains 109 NVFP4 linear layers;
- the 27-layer Qwen encoder contains 189 NVFP4 linear layers;
- dynamic activation-quantization metadata survived serialization.

The script fails instead of silently applying runtime weight quantization or
falling back to BF16.

## Rebuild The NVFP4 Artifact

The published artifact already exists, so this step is only required after
changing the source checkpoint, quantization configuration, or pinned software
versions.

```bash
modal run modal_prepare_nvfp4_artifact.py --force
```

The exporter:

1. Quantizes the transformer and active Qwen layers.
2. Saves to a staging directory on the Modal volume.
3. Reloads both components through their pre-quantized loaders.
4. Validates architecture, NVFP4 coverage, and activation configuration.
5. Atomically replaces the previous volume artifact.
6. Uploads the same validated files to Hugging Face.

The volume artifact is stored at:

```text
/mnt/klein4B-assets/FLUX.2-klein-4B-torchao-nvfp4
```

## Outputs

Generated images, prompts, and `benchmark.json` are written after timing to:

```text
/mnt/klein4B-assets/bench_outputs_nvfp4/<timestamp>/
```

The JSON report records wall and CUDA-event timing, compile configuration,
prompt-cache status, denoising-step validation, artifact revision, and NVFP4
layer coverage.

## Repository Layout

- [`klein_pipeline.py`](klein_pipeline.py): optimized Klein img2img pipeline and
  request-level caches.
- [`taef2_vae.py`](taef2_vae.py): TAEF2 integration.
- [`cache_dit_klein.py`](cache_dit_klein.py): attention selection and optional
  experimental runtime helpers.
- [`cuda_kernels/`](cuda_kernels/): experimental CUDA and Triton kernels.
- [`upload.py`](upload.py): original model and benchmark-image volume setup.
