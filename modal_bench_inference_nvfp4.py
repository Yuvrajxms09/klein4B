"""
Modal benchmark: same harness as modal_bench_inference.py but transformer loaded with
Diffusers TorchAoConfig + torchao NVFP4 (prototype mx_formats), not Photoroom FP8 static.

Requires a recent torchao (NVFP4 lives under torchao.prototype) and PyTorch new enough
for that config path; see torchao install docs if imports fail.
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
    # NVFP4 prototype + Diffusers TorchAo integration: prefer current torchao over a stale pin.
    .run_commands("pip install -U torchao")
    .add_local_dir(_repo_root(), remote_path="/root/klein4B")
    .add_local_dir(str(Path(_repo_root()).parent / "diffusers"), remote_path="/root/diffusers")
    .add_local_dir(str(Path(_repo_root()).parent / "flux2"), remote_path="/root/flux2")
    .add_local_dir(str(Path(_repo_root()).parent / "klein-cuda-c"), remote_path="/root/klein-cuda-c")
)


MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)
VOLUME_MOUNT = "/mnt/klein4B-assets"

DEFAULT_BENCHMARK_PROMPTS: tuple[str, ...] = (
    "on a top",
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
    print(f"nvfp4_transformer_loaded_from={model_dir} (subfolder=transformer)")
    return transformer


def _print_model_inventory(pipe) -> None:
    import torch

    transformer = getattr(pipe, "transformer", None)
    cfg = getattr(transformer, "config", None)
    qcfg = getattr(pipe, "quantization_config", None)
    if qcfg is None and cfg is not None:
        qcfg = getattr(cfg, "quantization_config", None)

    print(f"transformer_cls={type(transformer).__name__ if transformer is not None else None}")
    if cfg is not None:
        print(f"config_cls={type(cfg).__name__}")
        if hasattr(cfg, "to_dict"):
            try:
                cfg_dict = cfg.to_dict()
                print(f"config_keys={sorted(cfg_dict.keys())}")
                for key in sorted(cfg_dict.keys()):
                    value = cfg_dict[key]
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        print(f"config.{key}={value}")
            except Exception as exc:
                print(f"config_to_dict_failed={exc}")
        interesting = {
            "num_layers": getattr(cfg, "num_layers", None),
            "num_single_layers": getattr(cfg, "num_single_layers", None),
            "num_attention_heads": getattr(cfg, "num_attention_heads", None),
            "attention_head_dim": getattr(cfg, "attention_head_dim", None),
            "inner_dim": getattr(transformer, "inner_dim", None),
            "patch_size": getattr(cfg, "patch_size", None),
            "in_channels": getattr(cfg, "in_channels", None),
            "out_channels": getattr(cfg, "out_channels", None),
            "joint_attention_dim": getattr(cfg, "joint_attention_dim", None),
            "quantization_config_type": type(qcfg).__name__ if qcfg is not None else None,
            "quant_method": getattr(qcfg, "quant_method", None) if qcfg is not None else None,
            "activation_scheme": getattr(qcfg, "activation_scheme", None) if qcfg is not None else None,
            "weight_block_size": getattr(qcfg, "weight_block_size", None) if qcfg is not None else None,
            "use_mxfp8": getattr(qcfg, "use_mxfp8", None) if qcfg is not None else None,
        }
        for key, value in interesting.items():
            print(f"{key}={value}")

    if transformer is not None:
        print("transformer_top_level_children:")
        for name, module in transformer.named_children():
            print(f"  child={name} type={type(module).__name__}")

    linear_counts: dict[str, int] = {}
    quant_method_counts: dict[str, int] = {}
    total_linear = 0
    total_params = 0
    sample_names: list[str] = []
    for _, module in pipe.transformer.named_modules():
        total_params += sum(p.numel() for p in module.parameters(recurse=False))
        cls_name = type(module).__name__
        if "Linear" in cls_name:
            total_linear += 1
            linear_counts[cls_name] = linear_counts.get(cls_name, 0) + 1
            qm = getattr(module, "quant_method", None)
            qm_name = type(qm).__name__ if qm is not None else "None"
            quant_method_counts[qm_name] = quant_method_counts.get(qm_name, 0) + 1
        if len(sample_names) < 40:
            sample_names.append(cls_name)

    print(f"transformer_linear_modules={total_linear}")
    print(f"transformer_param_tensors={total_params}")
    print(f"linear_module_types={linear_counts}")
    print(f"linear_quant_methods={quant_method_counts}")
    print(f"module_type_sample={sample_names}")

    if transformer is not None:
        qkv_fused = 0
        attn_processors = set()
        attn_modules = []
        for name, module in transformer.named_modules():
            if hasattr(module, "processor"):
                attn_processors.add(type(module.processor).__name__)
                if len(attn_modules) < 30:
                    attn_modules.append((name, type(module).__name__, type(module.processor).__name__))
            if hasattr(module, "qkv") or hasattr(module, "to_qkv"):
                qkv_fused += 1
        print(f"attention_processor_types={sorted(attn_processors)}")
        print(f"attention_module_sample={attn_modules}")
        print(f"qkv_fused_module_count={qkv_fused}")
        print(f"torch_compile_default_dtype={torch.get_default_dtype()}")

    hf_cfg = getattr(pipe, "config", None)
    if hf_cfg is not None and hasattr(hf_cfg, "to_dict"):
        try:
            hf_dict = hf_cfg.to_dict()
            qcfg = hf_dict.get("quantization_config")
            print(f"pipe_config_keys={sorted(hf_dict.keys())}")
            print(f"pipe_quantization_config={qcfg}")
        except Exception as exc:
            print(f"pipe_config_to_dict_failed={exc}")


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
    print_model_inventory: bool = True,
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

    print("USE_TAEF2:", use_taef2)
    print("pipe.vae class:", type(pipe.vae).__name__)
    print("has taesd:", hasattr(pipe.vae, "taesd"))
    if hasattr(pipe.vae, "taesd"):
        print("taesd class:", type(pipe.vae.taesd).__name__)
        print("has taesd.encoder:", hasattr(pipe.vae.taesd, "encoder"))
        print("has taesd.decoder:", hasattr(pipe.vae.taesd, "decoder"))

    pipe.vae.to(memory_format=torch.channels_last)

    enable_cache_dit(pipe)
    backend = prepare_transformer_for_speed(pipe, backend="auto", fuse_qkv=True)
    print("Attention backend:", backend)

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

    print(type(pipe.transformer))
    print(type(pipe._vae_encode_fn))
    print(type(pipe._vae_decode_fn))
    print(hasattr(pipe.transformer, "_orig_mod"))
    print(hasattr(pipe._vae_encode_fn, "_torchdynamo_orig_callable"))
    print(hasattr(pipe._vae_decode_fn, "_torchdynamo_orig_callable"))

    if print_model_inventory:
        _print_model_inventory(pipe)

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
