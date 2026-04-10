from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import modal


APP = modal.App("klein4b-modal-cuda")


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
    .add_local_dir(_repo_root(), remote_path="/root/klein4B")
    .add_local_dir(str(Path(_repo_root()).parent / "diffusers"), remote_path="/root/diffusers")
    .add_local_dir(str(Path(_repo_root()).parent / "flux2"), remote_path="/root/flux2")
    .add_local_dir(str(Path(_repo_root()).parent / "klein-cuda-c"), remote_path="/root/klein-cuda-c")
)


MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)
VOLUME_MOUNT = "/mnt/klein4B-assets"


def _safe_prompt_slug(prompt: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^\w\s-]", "", prompt.lower())
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")[:max_len]
    return slug or "prompt"


def _build_cuda_extension() -> None:
    import torch
    from torch.utils.cpp_extension import load

    repo = Path("/root/klein4B/cuda_kernels")
    os.chdir(repo)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    load(
        name="klein_cuda_ext",
        sources=["src/ops.cpp", "src/ops_cuda.cu"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        is_python_module=False,
        verbose=True,
    )
    ns = torch.ops.klein_cuda
    caps = {
        "silu_mul_": hasattr(ns, "silu_mul_"),
        "adaln_norm": hasattr(ns, "adaln_norm"),
        "qk_rms_norm_": hasattr(ns, "qk_rms_norm_"),
        "rope_2d_offset_": hasattr(ns, "rope_2d_offset_"),
        "packed_attention_": hasattr(ns, "packed_attention_"),
        "fused_qkv_attention_": hasattr(ns, "fused_qkv_attention_"),
        "joint_packed_attention_": hasattr(ns, "joint_packed_attention_"),
    }
    print("klein_cuda_capabilities:")
    for name, enabled in caps.items():
        print(f"  {name}={enabled}")
    print("joint_packed_attention_status=", "ready" if caps["joint_packed_attention_"] else "rebuild_required")


def _load_nvfp4_transformer(*, model_dir: str, dtype, local_files_only: bool = True):
    import torch
    from diffusers import Flux2Transformer2DModel, TorchAoConfig

    try:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    except ImportError as exc:
        raise RuntimeError("NVFP4 import failed; install a recent torchao") from exc

    quantization_config = TorchAoConfig(
        NVFP4DynamicActivationNVFP4WeightConfig(
            use_triton_kernel=False,
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
    print(f"nvfp4_transformer_loaded_from={model_dir}")
    return transformer


def _save_images(*, images, prompts, base_dir: Path, volume: modal.Volume, volume_mount: str) -> Path:
    base_dir = base_dir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    for i, (prompt, img) in enumerate(zip(prompts, images), start=1):
        out_path = run_dir / f"{i:02d}_{_safe_prompt_slug(prompt)}.png"
        img.save(out_path, format="PNG")
    (run_dir / "prompts.txt").write_text("\n".join(f"{i:02d}\t{p}" for i, p in enumerate(prompts, start=1)) + "\n")
    if run_dir.resolve().is_relative_to(Path(volume_mount).resolve()):
        volume.commit()
    return run_dir


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
)
def build_and_run(
    *,
    model_dir: str = "/mnt/klein4B-assets/FLUX.2-klein-4B",
    image_path: str = "/mnt/klein4B-assets/calib/blue_car_resize.jpeg",
    height: int = 576,
    width: int = 384,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    generator_seed: int = 0,
    warmup_runs: int = 3,
    prompts: list[str] | None = None,
    use_taef2: bool = True,
    taef2_cache_dir: str = "/root/klein4B/.cache/taef2",
    save_outputs_dir: str | None = "/mnt/klein4B-assets/bench_outputs_cuda",
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

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    repo_root = Path("/root/klein4B")
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    diffusers_root = Path("/root/diffusers/src")
    if str(diffusers_root) not in sys.path:
        sys.path.insert(0, str(diffusers_root))
    flux2_root = Path("/root/flux2")
    if str(flux2_root) not in sys.path:
        sys.path.insert(0, str(flux2_root))

    from cache_dit_klein import prepare_transformer_for_speed, enable_cache_dit
    from klein_pipeline import Flux2KleinPipeline
    from taef2_vae import replace_pipeline_vae_with_taef2

    _build_cuda_extension()

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")

    prompt_list = list(prompts) if prompts is not None else ["a blue car on a mountain road at sunset"]
    input_image = Image.open(image_path).convert("RGB").resize((width, height))

    dtype = torch.bfloat16
    transformer = _load_nvfp4_transformer(model_dir=model_dir, dtype=dtype, local_files_only=local_files_only)
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
    backend = prepare_transformer_for_speed(pipe, backend="auto", fuse_qkv=True)
    print("Attention backend:", backend)

    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.epilogue_fusion = False

    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=False, dynamic=False)

    def _vae_encode_fn(image: torch.Tensor, generator: torch.Generator):
        return pipe._encode_vae_image(image=image, generator=generator)

    pipe._vae_encode_fn = torch.compile(_vae_encode_fn, mode="max-autotune", fullgraph=False, dynamic=False)

    def _vae_decode_fn(latents: torch.Tensor):
        return pipe.vae.decode(latents, return_dict=False)[0]

    pipe._vae_decode_fn = torch.compile(_vae_decode_fn, mode="max-autotune", fullgraph=False, dynamic=False)

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
    outputs = []
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

    print(f"First measured run: {times[0]:.1f} ms")
    print(f"Average over {len(times)} prompts: {sum(times) / len(times):.1f} ms")

    if save_outputs_dir:
        run_dir = _save_images(
            images=outputs,
            prompts=prompt_list,
            base_dir=Path(save_outputs_dir),
            volume=MODEL_VOLUME,
            volume_mount=VOLUME_MOUNT,
        )
        print(f"Saved {len(outputs)} images to {run_dir}")


@APP.local_entrypoint()
def main() -> None:
    build_and_run.remote()
