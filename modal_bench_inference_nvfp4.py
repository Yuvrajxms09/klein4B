"""
changes from last colab notebook:
1. using lighter vae (taef2)
2. nvfp4 weight + activation quantization at runtime
3. inductor flags for torch compile
4. different compile settings(using max-autotune this time)
5. some changes in transformer and ops in diffusers repo for a bit of speedup

## install dependencies 

## Reference dependency snapshot (Modal run, 2026-04-12)- works perfectly
- torch==2.11.0+cu130, torchao==0.17.0, Pillow==12.2.0, CUDA (PyTorch)==13.0

(might break because of diffusers or torch ao attribute error)
for all correct deps, please refer to modal image below but don't forget to install these 2 deps below:
- `pip install -U torchao` (NVFP4 needs `torchao.prototype.mx_formats`).
- `pip install -U mslk-cuda`

## setup — clone the required repos

1. klein4B with some new optimizations and fixed lighter vae (pls clone cuda-kernel branch)
- `git clone -b cuda-kernels --single-branch https://github.com/Yuvrajxms09/klein4B.git`

2. Diffusers with some optimizations to speed up ops (clone version2-flux2-speedups branch)
- `git clone -b version2-flux2-speedups --single-branch https://github.com/Yuvrajxms09/diffusers.git`

## NVFP4 weights + dynamic activations — `_load_nvfp4_transformer` (lines 156–182))

- Imports (inside `_load_nvfp4_transformer`): 
`import torch`; 
`from diffusers import Flux2Transformer2DModel, TorchAoConfig`;
`from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig`.

- Recreate `TorchAoConfig(NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=True, use_dynamic_per_tensor_scale=True))`
  and pass it to `Flux2Transformer2DModel.from_pretrained(..., subfolder="transformer", quantization_config=..., torch_dtype=dtype,
  local_files_only=...)`.

## CUDA matmul / TF32  (lines 242–251)

- torch.backends.cuda.matmul.allow_tf32 = True
- torch.backends.cudnn.allow_tf32 = True
- torch.set_float32_matmul_precision("high")

## Klein4B imports needed (lines 264–266)

- `from cache_dit_klein import prepare_transformer_for_speed, enable_cache_dit`
- `from klein_pipeline import Flux2KleinPipeline`
- `from taef2_vae import replace_pipeline_vae_with_taef2`

## Klein + DiT speed hooks + VAE layout(lines 294–306)

- TAEF2 branch: `replace_pipeline_vae_with_taef2(pipe, cache_dir=...)`; optional `pipe.vae.taesd.decoder.to(memory_format=torch.channels_last)`.
- Always run `pipe.vae.to(memory_format=torch.channels_last)` after the branch.
- Then `enable_cache_dit(pipe)` and `prepare_transformer_for_speed(pipe, backend="auto", fuse_qkv=True)`.

## torch.compile — Inductor flags - set them before using torch compile (lines 308–311)
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.epilogue_fusion = False

## torch.compile — transformer + VAE IO — `benchmark()` (lines 313–338)

- Assign `torch.compile(..., mode="max-autotune", fullgraph=False, dynamic=False)` to `pipe.transformer`, `pipe._vae_encode_fn`,
  and `pipe._vae_decode_fn`.
- Set all three `dynamic` values to `True` when height/width  changes between calls; keep `False` for fixed resolution
  (slightly faster- some ms for static shapes).

"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import modal


APP = modal.App("klein4b-inference-bench-nvfp4")


def _repo_root() -> str:
    return str(Path(__file__).resolve().parent)


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libopenblas-dev")
    .env({"TORCH_CUDA_ARCH_LIST": "8.9"})
    .pip_install("torch")
    .pip_install("torchvision")
    .pip_install("numpy")
    .pip_install("pillow")
    .pip_install("fastapi")
    .pip_install("uvicorn")
    .pip_install("accelerate")
    .pip_install("safetensors")
    .pip_install("huggingface_hub")
    .pip_install("transformers")
    .pip_install("einops")
    .pip_install("pybase64")
    .pip_install("cache-dit")
    .pip_install("ninja", "setuptools", "wheel")
    .run_commands("pip install -U torchao")
    .run_commands("pip install -U mslk-cuda")
    .add_local_dir(_repo_root(), remote_path="/root/klein4B")
    .add_local_dir(str(Path(_repo_root()).parent / "diffusers"), remote_path="/root/diffusers")
)


MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)
VOLUME_MOUNT = "/mnt/klein4B-assets"

DEFAULT_BENCHMARK_PROMPTS: tuple[str, ...] = (
    "from top angle",
    "on a bike",
    "on a mountain road",
    "in a rainy city at night",
    "under golden sunset light",
    "in a snowy forest",
    "with a cinematic teal and orange look",
    "as a watercolor painting",
    "as a realistic product photo",
    "with dramatic studio lighting",
    "in a futuristic cyberpunk street",
    "as a vintage film photograph",
    "with soft pastel colors",
    "as a high-detail oil painting",
    "in a desert at noon",
    "with neon reflections on wet pavement",
    "as a comic book illustration",
    "with a minimalist background",
    "in a lush tropical jungle",
    "as a high contrast black and white photo",
)


def _load_nvfp4_transformer(*, model_dir: str, dtype, local_files_only: bool = True):
    import torch
    from diffusers import Flux2Transformer2DModel, TorchAoConfig

    try:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    except ImportError as exc:
        raise RuntimeError(
            "NVFP4 import failed. Install a recent torchao (e.g. pip install -U torchao) "
            "and ensure torchao.prototype.mx_formats is available."
        ) from exc

    quantization_config = TorchAoConfig(
        NVFP4DynamicActivationNVFP4WeightConfig(
            use_triton_kernel=True,
            use_dynamic_per_tensor_scale=True,
        ),
    )

    transformer = Flux2Transformer2DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    return transformer


def _safe_prompt_slug(prompt: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^\w\s-]", "", prompt.lower())
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")[:max_len]
    return slug or "prompt"


def _save_benchmark_images_after_timing(
    *,
    images: list,
    prompts: list[str],
    base_dir: Path,
    volume: modal.Volume,
    volume_mount: str,
) -> Path:
    base_dir = base_dir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    for i, (prompt, img) in enumerate(zip(prompts, images), start=1):
        slug = _safe_prompt_slug(prompt)
        out_path = run_dir / f"{i:02d}_{slug}.png"
        img.save(out_path, format="PNG")

    lines = [f"{i:02d}\t{p}" for i, p in enumerate(prompts, start=1)]
    (run_dir / "prompts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if run_dir.resolve().is_relative_to(Path(volume_mount).resolve()):
        volume.commit()
    return run_dir


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
)
def benchmark(
    *,
    model_dir: str = "/mnt/klein4B-assets/FLUX.2-klein-4B",
    image_path: str = "/mnt/klein4B-assets/calib/blue_car.jpeg",
    height: int = 576,
    width: int = 384,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    generator_seed: int = 0,
    warmup_runs: int = 3,
    prompts: list[str] | None = None,
    use_taef2: bool = True,
    taef2_cache_dir: str = "/root/klein4B/.cache/taef2",
    save_outputs_dir: str | None = "/mnt/klein4B-assets/bench_outputs_nvfp4",
    local_files_only: bool = True,
) -> None:
    import torch
    from PIL import Image

    torch.set_grad_enabled(False)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    repo_root = Path("/root/klein4B")
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    diffusers_root = Path("/root/diffusers/src")
    if str(diffusers_root) not in sys.path:
        sys.path.insert(0, str(diffusers_root))

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    from cache_dit_klein import prepare_transformer_for_speed, enable_cache_dit
    from klein_pipeline import Flux2KleinPipeline
    from taef2_vae import replace_pipeline_vae_with_taef2

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")

    prompt_list = list(prompts) if prompts is not None else list(DEFAULT_BENCHMARK_PROMPTS)
    if not prompt_list:
        raise ValueError("prompts must be non-empty")

    input_image = Image.open(image_path).convert("RGB").resize((width, height))

    dtype = torch.bfloat16
    transformer = _load_nvfp4_transformer(
        model_dir=model_dir,
        dtype=dtype,
        local_files_only=local_files_only,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    if use_taef2:
        replace_pipeline_vae_with_taef2(pipe, cache_dir=taef2_cache_dir)
        if hasattr(pipe.vae, "taesd") and hasattr(pipe.vae.taesd, "decoder"):
            pipe.vae.taesd.decoder.to(memory_format=torch.channels_last)
    else:
        if hasattr(pipe.vae, "fuse_qkv_projections"):
            pipe.vae.fuse_qkv_projections()
        pipe.vae.to(memory_format=torch.channels_last)

    pipe.vae.to(memory_format=torch.channels_last)

    enable_cache_dit(pipe)
    prepare_transformer_for_speed(pipe, backend="auto", fuse_qkv=True)

    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.epilogue_fusion = False

    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
    )

    def _vae_encode_fn(image: torch.Tensor, generator: torch.Generator):
        return pipe._encode_vae_image(image=image, generator=generator)

    pipe._vae_encode_fn = torch.compile(
        _vae_encode_fn,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
    )

    def _vae_decode_fn(latents: torch.Tensor):
        return pipe.vae.decode(latents, return_dict=False)[0]

    pipe._vae_decode_fn = torch.compile(
        _vae_decode_fn,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
    )

    warmup_times: list[float] = []
    for i in range(warmup_runs):
        gen = torch.Generator(device="cuda").manual_seed(generator_seed)
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = pipe(
                prompt=prompt_list[0],
                image=input_image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=gen,
            ).images[0]
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        warmup_times.append(dt_ms)
        print(f"Warmup {i + 1}: {dt_ms:.1f} ms")

    print(f"Warmup avg: {sum(warmup_times) / len(warmup_times):.1f} ms")

    times: list[float] = []
    outputs: list[Image.Image] = []
    for i, prompt in enumerate(prompt_list, start=1):
        gen = torch.Generator(device="cuda").manual_seed(generator_seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            image_i2i = pipe(
                prompt=prompt,
                image=input_image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=gen,
            ).images[0]
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times.append(dt_ms)
        outputs.append(image_i2i)
        print(f"[{i:02d}] {dt_ms:.1f} ms | prompt: {prompt}")

    first_ms = times[0]
    avg_ms = sum(times) / len(times)
    print(f"\nFirst measured run: {first_ms:.1f} ms")
    print(f"Average over {len(times)} prompts: {avg_ms:.1f} ms")
    print(f"First output size (W,H): {outputs[0].size}")

    if save_outputs_dir:
        run_dir = _save_benchmark_images_after_timing(
            images=outputs,
            prompts=prompt_list,
            base_dir=Path(save_outputs_dir),
            volume=MODEL_VOLUME,
            volume_mount=VOLUME_MOUNT,
        )
        print(f"Saved {len(outputs)} images to {run_dir} (post-timed, not in latency)")


@APP.local_entrypoint()
def main(use_taef2: bool = True) -> None:
    benchmark.remote(use_taef2=use_taef2)
