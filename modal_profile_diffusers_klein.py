from __future__ import annotations

import copy
import functools
import gc
import logging
import os
import time
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import modal


APP = modal.App("klein4b-diffusers-profile")


def _repo_root() -> str:
    return str(Path(__file__).resolve().parent)


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "git-lfs", "libopenblas-dev")
    .env({"TORCH_CUDA_ARCH_LIST": "8.9"})
    .run_commands("git lfs install")
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
    .add_local_dir(str(Path(_repo_root()).parent / "flux2"), remote_path="/root/flux2")
    .add_local_dir(str(Path(_repo_root()).parent / "klein-cuda-c"), remote_path="/root/klein-cuda-c")
)


MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROMPT = "A blue car driving on a mountain road at sunset"
DEFAULT_IMAGE_PATH = "/mnt/klein4B-assets/calib/blue_car_resize.jpeg"


def _annotate(func, name):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import torch.profiler

        with torch.profiler.record_function(name):
            return func(*args, **kwargs)

    return wrapper


def _annotate_pipeline(pipe):
    for component_name, method_name, label in [
        ("transformer", "forward", "transformer_forward"),
        ("vae", "encode", "vae_encode"),
        ("vae", "decode", "vae_decode"),
        ("scheduler", "step", "scheduler_step"),
    ]:
        component = getattr(pipe, component_name, None)
        if component is None:
            continue
        method = getattr(component, method_name, None)
        if method is None:
            continue
        setattr(component, method_name, _annotate(method, label))

    if hasattr(pipe, "encode_prompt"):
        pipe.encode_prompt = _annotate(pipe.encode_prompt, "encode_prompt")


def _annotate_transformer_internals(pipe):
    import torch.profiler

    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return

    for name, module in transformer.named_modules():
        cls_name = type(module).__name__
        scope = None

        if "DoubleStreamBlock" in cls_name:
            scope = f"transformer.double_block.{name}"
        elif "SingleStreamBlock" in cls_name:
            scope = f"transformer.single_block.{name}"
        elif cls_name in {"SelfAttention", "Attention", "AttentionProcessor"}:
            scope = f"transformer.attention.{name}"
        elif cls_name in {"QKNorm", "RMSNorm", "LayerNorm"}:
            scope = f"transformer.norm.{name}"
        elif cls_name in {"Modulation", "MLPEmbedder", "SiLUActivation", "LastLayer"}:
            scope = f"transformer.misc.{name}"

        if scope is None:
            continue

        if hasattr(module, "forward"):
            module.forward = _annotate(module.forward, scope)

    try:
        import flux2.model as flux2_model

        if hasattr(flux2_model, "attention"):
            flux2_model.attention = _annotate(flux2_model.attention, "transformer.attention.fn")
        if hasattr(flux2_model, "apply_rope"):
            flux2_model.apply_rope = _annotate(flux2_model.apply_rope, "transformer.rope.fn")
        if hasattr(flux2_model, "_split_qkv_heads"):
            flux2_model._split_qkv_heads = _annotate(flux2_model._split_qkv_heads, "transformer.split_qkv.fn")
        if hasattr(flux2_model, "_gate_and_mlp"):
            flux2_model._gate_and_mlp = _annotate(flux2_model._gate_and_mlp, "transformer.gate_mlp.fn")
    except Exception:
        pass


def _flush():
    gc.collect()
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


@dataclass(frozen=True)
class ProfileConfig:
    model_dir: str = "/mnt/klein4B-assets/FLUX.2-klein-4B"
    image_path: str = DEFAULT_IMAGE_PATH
    prompt: str = PROMPT
    height: int = 576
    width: int = 384
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    seed: int = 0
    output_dir: str = "/mnt/klein4B-assets/profiling"
    output_type: str = "latent"
    compile_transformer: bool = False
    compile_mode: str = "max-autotune"
    compile_fullgraph: bool = False
    compile_regional: bool = True
    use_taef2: bool = True
    taef2_cache_dir: str = "/root/klein4B/.cache/taef2"


@dataclass(frozen=True)
class ProfileVariant:
    name: str
    cfg: ProfileConfig


def _load_image(path: str, width: int, height: int):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img = img.resize((width, height))
    return img


def _load_nvfp4_transformer(*, model_dir: str, dtype, local_files_only: bool = True):
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
    print(f"nvfp4_transformer_loaded_from={model_dir} (subfolder=transformer)")
    return transformer


def _profile_impl(cfg: ProfileConfig) -> str:
    import torch
    import torch.profiler
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

    transformer = _load_nvfp4_transformer(
        model_dir=cfg.model_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        cfg.model_dir,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if cfg.use_taef2:
        replace_pipeline_vae_with_taef2(pipe, cache_dir=cfg.taef2_cache_dir)
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

    _annotate_pipeline(pipe)
    _annotate_transformer_internals(pipe)

    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.epilogue_fusion = False

    if hasattr(pipe.transformer, "compile_repeated_blocks"):
        pipe.transformer.compile_repeated_blocks(
            mode=cfg.compile_mode,
            fullgraph=cfg.compile_fullgraph,
        )
    elif hasattr(pipe.transformer, "compile"):
        pipe.transformer.compile(
            mode=cfg.compile_mode,
            fullgraph=cfg.compile_fullgraph,
        )
    else:
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode=cfg.compile_mode,
            fullgraph=cfg.compile_fullgraph,
            dynamic=False,
        )

    def _vae_encode_fn(image: torch.Tensor, generator: torch.Generator):
        return pipe._encode_vae_image(image=image, generator=generator)

    pipe._vae_encode_fn = torch.compile(
        _vae_encode_fn,
        mode=cfg.compile_mode,
        fullgraph=False,
        dynamic=False,
    )

    def _vae_decode_fn(latents: torch.Tensor):
        return pipe.vae.decode(latents, return_dict=False)[0]

    pipe._vae_decode_fn = torch.compile(
        _vae_decode_fn,
        mode=cfg.compile_mode,
        fullgraph=False,
        dynamic=False,
    )

    img = _load_image(cfg.image_path, cfg.width, cfg.height)

    def run_once():
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
        return pipe(
            prompt=cfg.prompt,
            image=img,
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
            output_type=cfg.output_type,
            return_dict=True,
        )

    trace_dir = Path(cfg.output_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / f"klein4b_img2img_nvfp4_warm3_{'compile' if cfg.compile_transformer else 'eager'}_{cfg.output_type}.json"

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    warmup_times: list[float] = []
    prof = None
    for i in range(3):
        gen = torch.Generator(device="cuda").manual_seed(cfg.seed)
        t0 = time.perf_counter()
        if i == 2:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                with torch.profiler.record_function("pipeline_call"):
                    with torch.inference_mode():
                        _ = pipe(
                            prompt=cfg.prompt,
                            image=img,
                            height=cfg.height,
                            width=cfg.width,
                            num_inference_steps=cfg.num_inference_steps,
                            guidance_scale=cfg.guidance_scale,
                            generator=gen,
                            output_type=cfg.output_type,
                            return_dict=True,
                        )
        else:
            with torch.inference_mode():
                _ = pipe(
                    prompt=cfg.prompt,
                    image=img,
                    height=cfg.height,
                    width=cfg.width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    generator=gen,
                    output_type=cfg.output_type,
                    return_dict=True,
                )
        torch.cuda.synchronize()
        warmup_times.append((time.perf_counter() - t0) * 1000.0)
        print(f"Warmup {i + 1}: {warmup_times[-1]:.1f} ms")

    if prof is None:
        raise RuntimeError("Profiler did not run on warmup 3")

    prof.export_chrome_trace(str(trace_file))
    print(f"Warmup avg: {sum(warmup_times) / len(warmup_times):.1f} ms")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
    print(f"trace_file={trace_file}")

    _flush()
    return str(trace_file)


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60,
    volumes={"/mnt/klein4B-assets": MODEL_VOLUME},
)
def run_profile(
    cfg: ProfileConfig = ProfileConfig(),
) -> str:
    return _profile_impl(cfg)


@APP.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/mnt/klein4B-assets": MODEL_VOLUME},
)
def run_profiles(
    variants: list[ProfileVariant],
) -> list[str]:
    results = []
    for variant in variants:
        logger.info("Running profile variant: %s", variant.name)
        result = _profile_impl(variant.cfg)
        results.append(result)
    return results


def _build_default_variants() -> list[ProfileVariant]:
    base = dict(
        model_dir=os.environ.get("KLEIN4B_MODEL_DIR", ProfileConfig.model_dir),
        image_path=os.environ.get("KLEIN4B_IMAGE_PATH", ProfileConfig.image_path),
        prompt=os.environ.get("KLEIN4B_PROMPT", ProfileConfig.prompt),
        height=int(os.environ.get("KLEIN4B_HEIGHT", ProfileConfig.height)),
        width=int(os.environ.get("KLEIN4B_WIDTH", ProfileConfig.width)),
        num_inference_steps=int(os.environ.get("KLEIN4B_STEPS", ProfileConfig.num_inference_steps)),
        guidance_scale=float(os.environ.get("KLEIN4B_GUIDANCE", ProfileConfig.guidance_scale)),
        seed=int(os.environ.get("KLEIN4B_SEED", ProfileConfig.seed)),
        output_dir=os.environ.get("KLEIN4B_OUTPUT_DIR", ProfileConfig.output_dir),
        compile_mode=os.environ.get("KLEIN4B_COMPILE_MODE", ProfileConfig.compile_mode),
    )

    return [
        ProfileVariant(
            name="eager_latent",
            cfg=ProfileConfig(**base, output_type="latent", compile_transformer=False, compile_fullgraph=False, compile_regional=True),
        ),
        ProfileVariant(
            name="compile_latent_regional",
            cfg=ProfileConfig(**base, output_type="latent", compile_transformer=True, compile_fullgraph=False, compile_regional=True),
        ),
        ProfileVariant(
            name="compile_latent_fullgraph",
            cfg=ProfileConfig(**base, output_type="latent", compile_transformer=True, compile_fullgraph=True, compile_regional=False),
        ),
        ProfileVariant(
            name="eager_pil",
            cfg=ProfileConfig(**base, output_type="pil", compile_transformer=False, compile_fullgraph=False, compile_regional=True),
        ),
        ProfileVariant(
            name="compile_pil_regional",
            cfg=ProfileConfig(**base, output_type="pil", compile_transformer=True, compile_fullgraph=False, compile_regional=True),
        ),
    ]


@APP.local_entrypoint()
def main():
    variants = _build_default_variants()

    futures = {}
    with ThreadPoolExecutor(max_workers=len(variants)) as executor:
        for variant in variants:
            futures[executor.submit(run_profile.remote, variant.cfg)] = variant.name

        results = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                logger.exception("Variant failed: %s", name)
                results[name] = {"error": str(exc)}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
