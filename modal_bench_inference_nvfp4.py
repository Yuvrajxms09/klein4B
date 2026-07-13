"""
changes from last colab notebook:
1. using lighter vae (taef2)
2. nvfp4 weight + activation quantization at runtime
3. inductor flags for torch compile
4. different compile settings(using max-autotune this time)
5. some changes in transformer and ops in diffusers repo for a bit of speedup

## install dependencies 

## Reference dependency snapshot
- torch>=2.11 with CUDA 13, local `ao/` checkout (0.18.0), Pillow>=12

(might break because of diffusers or torch ao attribute error)
for all correct deps, please refer to modal image below but don't forget to install these 2 deps below:
- Install the bundled local `ao/` checkout (NVFP4 needs `torchao.prototype.mx_formats`).
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
import json
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import modal


APP = modal.App("klein4b-inference-bench-nvfp4")


def _repo_root() -> str:
    return str(Path(__file__).resolve().parent)


CUTLASS_INCLUDE_SOURCE = (
    Path(_repo_root()).parent / "F2K_CUDA" / "third_party" / "cutlass" / "include"
)
CUTLASS_UTIL_INCLUDE_SOURCE = (
    Path(_repo_root()).parent
    / "F2K_CUDA"
    / "third_party"
    / "cutlass"
    / "tools"
    / "util"
    / "include"
)


image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libopenblas-dev")
    .env({"TORCH_CUDA_ARCH_LIST": "12.0"})
    .run_commands(
        "pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130"
    )
    .pip_install("numpy")
    .pip_install("pillow")
    .pip_install("fastapi")
    .pip_install("uvicorn")
    .pip_install("accelerate")
    .pip_install("safetensors")
    .pip_install("omegaconf>=2.3.0", "pulp<4.0", "rich", "nvidia-ml-py>=12", "PyYAML>=6.0")
    .pip_install(
        "scipy",
        "datasets>=3.0.0",
        "nltk",
        "peft>=0.17.0",
        "sentencepiece>=0.2.1",
        "tiktoken",
        "wonderwords",
    )
    .pip_install("huggingface_hub")
    .pip_install("transformers>=4.56,<5.13")
    .pip_install("compressed-tensors>=0.15.0")
    .pip_install("einops")
    .pip_install("pybase64")
    .pip_install("cache-dit")
    .pip_install("flashinfer-python==0.6.14")
    .pip_install(
        "ninja",
        "setuptools>=80",
        "setuptools-scm>=8,<10",
        "wheel",
        "packaging",
        "pydantic>=2.0",
        "regex",
        "tqdm",
    )
    .run_commands(
        "pip install mslk-cuda==1.1.0",
        "python -c \"import torch, mslk; assert torch.__version__.startswith('2.11.'); print('mslk_abi_ok', torch.__version__, torch.version.cuda, mslk.__version__)\"",
    )
    .add_local_dir(_repo_root(), remote_path="/root/klein4B", copy=True)
    .add_local_dir(
        str(Path(_repo_root()).parent / "diffusers"),
        remote_path="/root/diffusers",
        copy=True,
    )
    .add_local_dir(
        str(Path(_repo_root()).parent / "ao"),
        remote_path="/root/ao",
        copy=True,
    )
    .add_local_dir(
        str(Path(_repo_root()).parent / "Model-Optimizer"),
        remote_path="/root/Model-Optimizer",
        copy=True,
    )
    # TorchAO's setup.py imports torch at build time; its pyproject explicitly
    # requires installation without PEP 517 build isolation.
    .run_commands(
        "USE_CPP=0 pip install --no-build-isolation --no-deps -e /root/ao"
    )
    # ModelOpt restore needs its core dependencies only. Its full HF extra also
    # installs DeepSpeed and a PyPI Diffusers build, which are unnecessary here
    # and can conflict with the checked-out Diffusers source inserted at runtime.
    .run_commands(
        "pip install --no-build-isolation --no-deps -e /root/Model-Optimizer"
    )
)
if CUTLASS_INCLUDE_SOURCE.is_dir():
    image = image.add_local_dir(
        str(CUTLASS_INCLUDE_SOURCE),
        remote_path="/root/cutlass/include",
        copy=True,
    )
if CUTLASS_UTIL_INCLUDE_SOURCE.is_dir():
    image = image.add_local_dir(
        str(CUTLASS_UTIL_INCLUDE_SOURCE),
        remote_path="/root/cutlass/tools/util/include",
        copy=True,
    )


MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)
VOLUME_MOUNT = "/mnt/klein4B-assets"


def _build_klein_cuda_extension(*, required: bool) -> bool:
    """Build and register the local CUDA operators for an explicit A/B run."""
    import torch
    from torch.utils.cpp_extension import load

    if torch.version.cuda is None:
        if required:
            raise RuntimeError("Klein CUDA operators require a CUDA-enabled PyTorch build")
        return False

    cuda_major = int(torch.version.cuda.split(".", maxsplit=1)[0])
    if cuda_major < 13:
        message = (
            f"Klein CUDA operators target SM120 NVFP4 and require CUDA 13+, "
            f"but this image provides CUDA {torch.version.cuda}"
        )
        if required:
            raise RuntimeError(message)
        print(f"klein_cuda_ops=disabled reason={message}")
        return False

    repo = Path("/root/klein4B/cuda_kernels")
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo)
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
        load(
            name="klein_cuda_ext",
            sources=["src/ops.cpp", "src/ops_cuda.cu"],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            is_python_module=False,
            verbose=True,
        )
    finally:
        os.chdir(previous_cwd)

    ns = torch.ops.klein_cuda
    required_ops = ("adaln_norm", "qk_rms_norm_", "rope_2d_offset_", "silu_mul_")
    missing = [name for name in required_ops if not hasattr(ns, name)]
    if missing:
        raise RuntimeError(f"Klein CUDA extension loaded without required operators: {missing}")
    print(f"klein_cuda_ops=loaded torch_cuda={torch.version.cuda} arch={os.environ['TORCH_CUDA_ARCH_LIST']}")
    return True


def _build_sm120_nvfp4_extension():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if torch.version.cuda is None or int(torch.version.cuda.split(".", maxsplit=1)[0]) < 13:
        raise RuntimeError(f"SM120 CUTLASS NVFP4 requires CUDA 13+, got {torch.version.cuda}")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(f"SM120 CUTLASS NVFP4 requires compute capability 12.0, got {capability}")
    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
        raise RuntimeError(f"SM120 CUTLASS build requires NVCC, CUDA_HOME={CUDA_HOME!r}")

    source_dir = Path("/root/klein4B/cuda_kernels")
    cutlass_version_header = Path("/root/cutlass/include/cutlass/version.h")
    cutlass_util_include = Path("/root/cutlass/tools/util/include")
    packed_stride_header = cutlass_util_include / "cutlass" / "util" / "packed_stride.hpp"
    if not cutlass_version_header.is_file():
        raise FileNotFoundError(f"CUTLASS version header not found: {cutlass_version_header}")
    if not packed_stride_header.is_file():
        raise FileNotFoundError(
            "CUTLASS utility headers were not copied into the image: "
            f"missing={packed_stride_header}"
        )
    version_text = cutlass_version_header.read_text(encoding="utf-8")

    def version_component(name: str) -> int:
        match = re.search(rf"^#define CUTLASS_{name} (\d+)$", version_text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"CUTLASS version header is missing CUTLASS_{name}")
        return int(match.group(1))

    cutlass_version = tuple(version_component(name) for name in ("MAJOR", "MINOR", "PATCH"))
    if cutlass_version != (4, 5, 2):
        raise RuntimeError(
            "SM120 NVFP4 extension is pinned to CUTLASS 4.5.2, "
            f"found={'.'.join(map(str, cutlass_version))}"
        )
    print(
        "sm120_nvfp4_extension=build_start "
        f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
        f"cuda_home={CUDA_HOME} cutlass={'.'.join(map(str, cutlass_version))} "
        "nvcc_arch=sm_120f"
    )
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    try:
        extension = load(
            name="klein_sm120_nvfp4_ext",
            sources=[
                str(source_dir / "sm120_nvfp4.cpp"),
                str(source_dir / "sm120_nvfp4.cu"),
            ],
            extra_include_paths=["/root/cutlass/include", str(cutlass_util_include)],
            extra_cflags=["-O3", "-std=c++17"],
            # PyTorch does not accept the family suffix in TORCH_CUDA_ARCH_LIST.
            # An explicit NVCC arch flag suppresses its generated plain-sm120 flag.
            extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_120f"],
            with_cuda=True,
            verbose=True,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch

    from cuda_kernels.sm120_nvfp4 import register_extension

    register_extension(extension)
    if not hasattr(torch.ops.klein_sm120, "nvfp4_gemm_out"):
        raise RuntimeError("SM120 extension loaded without nvfp4_gemm_out")
    dispatch_checks = {
        key: bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "klein_sm120::nvfp4_gemm_out", key
            )
        )
        for key in ("CUDA", "Meta")
    }
    if not all(dispatch_checks.values()):
        raise RuntimeError(f"SM120 extension dispatch registration incomplete: {dispatch_checks}")
    print(
        "sm120_nvfp4_extension=loaded "
        "tiles=128x128x256,128x128x128,128x64x256,128x64x128,"
        "128x32x256,128x32x128 cluster=1x1x1 "
        "epilogue=dynamic-scale workspace=zero bias=fallback-to-torchao "
        f"dispatch={json.dumps(dispatch_checks, sort_keys=True)} "
        "selection=per-production-shape"
    )
    return extension

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

STATIC_ACTIVATION_CALIBRATION_PROMPTS: tuple[str, ...] = (
    "calibration view from a low camera angle",
    "calibration scene under overcast daylight",
    "calibration edit with reflective metallic surfaces",
    "calibration image in a detailed indoor environment",
    "calibration scene with bright neon highlights and deep shadows",
    "calibration close-up with fine fabric and skin textures",
    "calibration landscape containing snow water and dark trees",
    "calibration product photograph on a clean studio background",
)

WARMUP_PROMPT = "warmup-only prompt not used in measured results"
STATIC_ACTIVATION_PARITY_PROMPT = "held-out static activation parity scene"


def _load_nvfp4_transformer(
    *,
    model_dir: str,
    dtype,
    local_files_only: bool = True,
    fuse_qkv_before_quantization: bool = True,
):
    import torch
    from diffusers import Flux2Transformer2DModel, TorchAoConfig

    try:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    except ImportError as exc:
        raise RuntimeError(
            "NVFP4 import failed. Install a recent torchao (e.g. pip install -U torchao) "
            "and ensure torchao.prototype.mx_formats is available."
        ) from exc

    nvfp4_config = NVFP4DynamicActivationNVFP4WeightConfig(
        use_triton_kernel=True,
        use_dynamic_per_tensor_scale=True,
    )

    if fuse_qkv_before_quantization:
        from torchao.quantization import quantize_

        transformer = Flux2Transformer2DModel.from_pretrained(
            model_dir,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to("cuda")

        expected_fused_modules = len(transformer.transformer_blocks)
        transformer.fuse_qkv_projections()
        transformer._is_qkv_fused = True
        fused_modules = [
            module
            for module in transformer.modules()
            if module.__class__.__name__ == "Flux2Attention" and getattr(module, "fused_projections", False)
        ]
        if len(fused_modules) != expected_fused_modules:
            raise RuntimeError(
                "BF16-first QKV fusion did not cover every double-stream block: "
                f"expected={expected_fused_modules}, observed={len(fused_modules)}"
            )
        for index, module in enumerate(fused_modules):
            if not hasattr(module, "to_qkv") or not hasattr(module, "to_added_qkv"):
                raise RuntimeError(f"fused Flux2Attention module {index} is missing a fused projection")

        inactive_projection_ids = {
            id(projection)
            for module in fused_modules
            for projection in (
                module.to_q,
                module.to_k,
                module.to_v,
                module.add_q_proj,
                module.add_k_proj,
                module.add_v_proj,
            )
        }

        def active_linear_filter(module: torch.nn.Module, _fqn: str) -> bool:
            return isinstance(module, torch.nn.Linear) and id(module) not in inactive_projection_ids

        quantize_(transformer, nvfp4_config, filter_fn=active_linear_filter)
        nvfp4_linear_count = sum(
            1
            for module in transformer.modules()
            if isinstance(module, torch.nn.Linear) and module.weight.__class__.__name__ == "NVFP4Tensor"
        )
        expected_active_nvfp4_linears = 109 - 4 * len(fused_modules)
        if nvfp4_linear_count != expected_active_nvfp4_linears:
            raise RuntimeError(
                "TorchAO quantized an unexpected number of active fused transformer linears: "
                f"expected={expected_active_nvfp4_linears}, observed={nvfp4_linear_count}"
            )

        projection_validation = []
        generator = torch.Generator(device="cuda").manual_seed(0)
        with torch.inference_mode():
            for block_index, module in enumerate(fused_modules):
                sample = torch.randn(
                    (128, module.to_q.in_features),
                    generator=generator,
                    device="cuda",
                    dtype=dtype,
                )
                for stream, fused, separate in (
                    ("image", module.to_qkv, (module.to_q, module.to_k, module.to_v)),
                    (
                        "text",
                        module.to_added_qkv,
                        (module.add_q_proj, module.add_k_proj, module.add_v_proj),
                    ),
                ):
                    bf16_reference = torch.cat([projection(sample) for projection in separate], dim=-1).float()
                    nvfp4_reference_parts = []
                    for projection in separate:
                        quantized_projection = torch.nn.Linear(
                            projection.in_features,
                            projection.out_features,
                            bias=projection.bias is not None,
                            device=projection.weight.device,
                            dtype=projection.weight.dtype,
                        )
                        quantized_projection.load_state_dict(projection.state_dict())
                        quantize_(quantized_projection, nvfp4_config)
                        if quantized_projection.weight.__class__.__name__ != "NVFP4Tensor":
                            raise RuntimeError(
                                "TorchAO did not quantize the temporary QKV reference projection"
                            )
                        nvfp4_reference_parts.append(quantized_projection(sample))
                    nvfp4_reference = torch.cat(nvfp4_reference_parts, dim=-1).float()
                    candidate = fused(sample).float()
                    relative_l2 = float(
                        (candidate - nvfp4_reference).norm()
                        / nvfp4_reference.norm().clamp_min(1e-12)
                    )
                    cosine = float(
                        torch.nn.functional.cosine_similarity(
                            candidate.flatten(), nvfp4_reference.flatten(), dim=0
                        )
                    )
                    bf16_relative_l2 = float(
                        (candidate - bf16_reference).norm()
                        / bf16_reference.norm().clamp_min(1e-12)
                    )
                    bf16_cosine = float(
                        torch.nn.functional.cosine_similarity(
                            candidate.flatten(), bf16_reference.flatten(), dim=0
                        )
                    )
                    item = {
                        "block": block_index,
                        "stream": stream,
                        "incremental_relative_l2": relative_l2,
                        "incremental_cosine": cosine,
                        "bf16_relative_l2": bf16_relative_l2,
                        "bf16_cosine": bf16_cosine,
                    }
                    projection_validation.append(item)

        rejected_projections = [
            item
            for item in projection_validation
            if item["incremental_relative_l2"] > 0.1 or item["incremental_cosine"] < 0.99
        ]
        if rejected_projections:
            transformer.unfuse_qkv_projections()
            transformer._is_qkv_fused = False

            def original_projection_filter(module: torch.nn.Module, _fqn: str) -> bool:
                return isinstance(module, torch.nn.Linear) and id(module) in inactive_projection_ids

            quantize_(transformer, nvfp4_config, filter_fn=original_projection_filter)
            nvfp4_linear_count = sum(
                1
                for module in transformer.modules()
                if isinstance(module, torch.nn.Linear) and module.weight.__class__.__name__ == "NVFP4Tensor"
            )
            if nvfp4_linear_count != 109:
                raise RuntimeError(
                    "QKV fusion fallback produced an unexpected NVFP4 topology: "
                    f"expected=109 observed={nvfp4_linear_count}"
                )
            fusion_report = {
                "requested": True,
                "enabled": False,
                "fallback": "separate-torchao-nvfp4-projections",
                "rejected_projection_count": len(rejected_projections),
                "fused_double_blocks": 0,
                "nvfp4_linear_modules": nvfp4_linear_count,
                "expected_nvfp4_gemms_per_step": 109,
                "projection_validation": projection_validation,
            }
            transformer._klein_prequant_qkv_fusion_report = fusion_report
            print(f"prequant_qkv_fusion=rejected report={json.dumps(fusion_report, sort_keys=True)}")
            return transformer

        # Six Q/K/V projections become two per double block. The original
        # full-compute Klein path executes 109 NVFP4 GEMMs per denoising step.
        gemms_per_step = 109 - 4 * len(fused_modules)
        fusion_report = {
            "requested": True,
            "enabled": True,
            "fallback": None,
            "fused_double_blocks": len(fused_modules),
            "nvfp4_linear_modules": nvfp4_linear_count,
            "inactive_bf16_fallback_projections": len(inactive_projection_ids),
            "expected_nvfp4_gemms_per_step": gemms_per_step,
            "projection_validation": projection_validation,
        }
        transformer._klein_prequant_qkv_fusion_report = fusion_report
        print(f"prequant_qkv_fusion={json.dumps(fusion_report, sort_keys=True)}")
    else:
        quantization_config = TorchAoConfig(nvfp4_config)
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_dir,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        transformer._klein_prequant_qkv_fusion_report = {
            "requested": False,
            "enabled": False,
            "fallback": None,
            "fused_double_blocks": 0,
            "expected_nvfp4_gemms_per_step": 109,
        }
    return transformer


def _load_modelopt_nvfp4_transformer(
    *,
    model_dir: str,
    state_path: str,
    dtype,
    local_files_only: bool = True,
):
    import modelopt.torch.opt as mto
    import torch
    from diffusers import Flux2Transformer2DModel

    state_file = Path(state_path)
    if not state_file.is_file():
        raise FileNotFoundError(f"ModelOpt state not found: {state_file}")

    transformer = Flux2Transformer2DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    transformer = mto.restore(transformer, str(state_file))
    if not isinstance(transformer, torch.nn.Module):
        raise TypeError(f"ModelOpt restore returned {type(transformer)!r}, expected nn.Module")
    print(f"modelopt_nvfp4_restored state={state_file}")
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
    benchmark_result: dict | None = None,
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
    if benchmark_result is not None:
        (run_dir / "benchmark.json").write_text(
            json.dumps(benchmark_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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
    warmup_runs: int = 5,
    measured_runs: int = 20,
    prompts: list[str] | None = None,
    use_taef2: bool = True,
    taef2_cache_dir: str = "/root/klein4B/.cache/taef2",
    save_outputs_dir: str | None = "/mnt/klein4B-assets/bench_outputs_nvfp4",
    local_files_only: bool = True,
    output_type: str = "pil",
    enable_cache: bool = False,
    cache_steps_mask: str = "1111",
    cache_steps_computation_policy: str = "dynamic",
    cache_residual_diff_threshold: float = 0.8,
    cache_single_block_rdt_scale: float = 3.0,
    require_cache_reuse_candidate: bool = False,
    backend: str = "torchao-nvfp4",
    modelopt_state_path: str = "/mnt/klein4B-assets/quantized_klein4b/nvfp4_hf/transformer_modelopt_state.pt",
    enable_klein_cuda_ops: bool = False,
    enable_direct_nvfp4_dispatch: bool = True,
    enable_static_transformer_activation_scales: bool = False,
    enable_full_denoise_compile: bool = False,
    enable_denoiser_step_reuse: bool = False,
    fuse_qkv_before_quantization: bool = False,
    enable_fused_qkv_packing: bool = False,
    nvfp4_gemm_backend: str = "torch-scaled-mm",
    attention_backend_request: str = "auto",
    text_encoder_backend: str = "bf16",
    nvfp4_text_encoder_dir: str = "/mnt/klein4B-assets/Qwen3-4B-NVFP4",
    compile_text_encoder: bool = True,
    enable_torch_compile: bool = True,
    overlap_preparation: bool = False,
    optimization_profile: str = "baseline",
    nsight_capture: bool = False,
    enable_internal_profiler_validation: bool = True,
) -> dict:
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

    from cache_dit_klein import (
        calibrate_torchao_nvfp4_activation_scales,
        enable_cache_dit,
        enable_compiled_denoise_loop,
        prepare_transformer_for_speed,
    )
    from klein_pipeline import Flux2KleinPipeline
    from taef2_vae import replace_pipeline_vae_with_taef2

    supported_profiles = {
        "baseline",
        "one-shot-exact",
        "one-shot-fast",
        "one-shot-aggressive",
        "one-shot-80",
        "one-shot-80-quality",
    }
    if optimization_profile not in supported_profiles:
        raise ValueError(f"Unsupported optimization_profile: {optimization_profile!r}")
    if optimization_profile in {
        "one-shot-exact",
        "one-shot-fast",
        "one-shot-aggressive",
        "one-shot-80",
        "one-shot-80-quality",
    }:
        # Share the exact kernel stack. The fast profile additionally enables
        # one explicit final-step reuse policy below.
        backend = "torchao-nvfp4"
        text_encoder_backend = "torchao-nvfp4"
        use_taef2 = True
        output_type = "pil"
        enable_cache = False
        enable_denoiser_step_reuse = optimization_profile in {"one-shot-80", "one-shot-80-quality"}
        enable_klein_cuda_ops = False
        enable_direct_nvfp4_dispatch = True
        enable_full_denoise_compile = False
        fuse_qkv_before_quantization = True
        enable_fused_qkv_packing = True
        nvfp4_gemm_backend = (
            "torch-scaled-mm" if optimization_profile == "one-shot-80-quality" else "cutlass-sm120"
        )
        attention_backend_request = "auto"
        enable_torch_compile = optimization_profile != "one-shot-80-quality"
        compile_text_encoder = enable_torch_compile
        overlap_preparation = True
        allow_approximate_attention = optimization_profile in {
            "one-shot-fast",
            "one-shot-aggressive",
            "one-shot-80",
            "one-shot-80-quality",
        }
        if optimization_profile in {"one-shot-fast", "one-shot-aggressive"}:
            enable_cache = True
            cache_steps_mask = "1110" if optimization_profile == "one-shot-fast" else "1100"
            cache_steps_computation_policy = "static"
            require_cache_reuse_candidate = True
        elif enable_denoiser_step_reuse:
            cache_steps_mask = "1100"
        print(
            f"optimization_profile={optimization_profile} "
            "transformer=torchao-nvfp4 text_encoder=torchao-nvfp4 "
            "qkv_fusion=bf16-before-quantization triton_packing=true "
            f"direct_dispatch=true gemm={nvfp4_gemm_backend} attention=auto "
            "preparation_overlap=true output=pil "
            f"cache={enable_cache} denoiser_step_reuse={enable_denoiser_step_reuse} "
            f"steps_mask={cache_steps_mask} "
            f"cache_policy={cache_steps_computation_policy}"
            f" approximate_attention={allow_approximate_attention}"
            f" torch_compile={enable_torch_compile}"
        )
        if (height, width) != (576, 384):
            raise ValueError(
                f"{optimization_profile} is shape-specialized for height=576 width=384, "
                f"got height={height} width={width}"
            )
        if num_inference_steps != 4 or guidance_scale != 1.0:
            raise ValueError(
                f"{optimization_profile} requires four steps and guidance_scale=1.0, "
                f"got steps={num_inference_steps} guidance_scale={guidance_scale}"
            )

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")

    prompt_list = list(prompts) if prompts is not None else list(DEFAULT_BENCHMARK_PROMPTS)
    if not prompt_list:
        raise ValueError("prompts must be non-empty")
    if len(prompt_list) != 20 or len(set(prompt_list)) != 20:
        raise ValueError("benchmark contract requires exactly 20 distinct prompts")
    if enable_fused_qkv_packing and backend != "torchao-nvfp4":
        raise ValueError("fused QKV packing is supported only by the TorchAO NVFP4 path")
    if enable_fused_qkv_packing and not fuse_qkv_before_quantization:
        raise ValueError("fused QKV packing requires BF16-first QKV fusion")
    if enable_fused_qkv_packing and enable_klein_cuda_ops:
        raise ValueError("Qwen-style Triton packing and the experimental C block patch are separate A/B tracks")
    if text_encoder_backend not in {"bf16", "torchao-nvfp4", "compressed-tensors-nvfp4"}:
        raise ValueError(f"Unsupported text_encoder_backend: {text_encoder_backend!r}")
    if cache_steps_computation_policy not in {"dynamic", "static"}:
        raise ValueError(
            "cache_steps_computation_policy must be 'dynamic' or 'static', "
            f"got {cache_steps_computation_policy!r}"
        )
    if text_encoder_backend == "compressed-tensors-nvfp4" and not Path(nvfp4_text_encoder_dir).is_dir():
        raise FileNotFoundError(f"NVFP4 text encoder directory not found: {nvfp4_text_encoder_dir}")

    input_image = Image.open(image_path).convert("RGB").resize((width, height))

    dtype = torch.bfloat16
    if backend == "torchao-nvfp4":
        transformer = _load_nvfp4_transformer(
            model_dir=model_dir,
            dtype=dtype,
            local_files_only=local_files_only,
            fuse_qkv_before_quantization=fuse_qkv_before_quantization,
        )
        transformer_fusion_report = transformer._klein_prequant_qkv_fusion_report
        if fuse_qkv_before_quantization and not transformer_fusion_report["enabled"]:
            if optimization_profile in {
                "one-shot-fast",
                "one-shot-aggressive",
                "one-shot-80",
                "one-shot-80-quality",
            }:
                raise RuntimeError(
                    f"{optimization_profile} requires the fused 89-linear topology; "
                    f"QKV fusion failed: {transformer_fusion_report['fallback']}"
                )
            fuse_qkv_before_quantization = False
            enable_fused_qkv_packing = False
            print(
                "one_shot_qkv_fusion=fallback "
                f"reason={transformer_fusion_report['fallback']} "
                "triton_fused_qkv_packing=disabled"
            )
    elif backend == "modelopt-nvfp4":
        sys.path.insert(0, "/root/Model-Optimizer")
        transformer = _load_modelopt_nvfp4_transformer(
            model_dir=model_dir,
            state_path=modelopt_state_path,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend!r}")

    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    text_encoder_validation = None
    if text_encoder_backend in {"bf16", "torchao-nvfp4"}:
        validation_prompts = prompt_list[:3]
        pipe._prompt_cache.clear()
        with torch.inference_mode():
            reference_embeddings = [
                pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)[0].detach().clone()
                for prompt in validation_prompts
            ]

        decoder = pipe.text_encoder.model
        original_layer_count = len(decoder.layers)
        required_layer_count = 27
        if original_layer_count < required_layer_count:
            raise RuntimeError(
                f"Qwen text encoder has {original_layer_count} layers; Klein requires hidden state 27"
            )
        decoder.layers = torch.nn.ModuleList(list(decoder.layers[:required_layer_count]))

        nvfp4_linear_count = 0
        if text_encoder_backend == "torchao-nvfp4":
            from torchao.quantization import quantize_
            from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

            quantize_(
                decoder,
                NVFP4DynamicActivationNVFP4WeightConfig(
                    use_triton_kernel=True,
                    use_dynamic_per_tensor_scale=True,
                ),
                device="cuda",
            )
            nvfp4_linear_count = sum(
                1
                for module in decoder.modules()
                if isinstance(module, torch.nn.Linear)
                and module.weight.__class__.__name__ == "NVFP4Tensor"
            )
            expected_linear_count = required_layer_count * 7
            if nvfp4_linear_count != expected_linear_count:
                raise RuntimeError(
                    "Unexpected Qwen NVFP4 coverage: "
                    f"expected={expected_linear_count}, observed={nvfp4_linear_count}"
                )

        pipe._prompt_cache.clear()
        with torch.inference_mode():
            candidate_embeddings = [
                pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)[0].detach().clone()
                for prompt in validation_prompts
            ]
        parity = []
        for prompt, reference, candidate in zip(
            validation_prompts, reference_embeddings, candidate_embeddings, strict=True
        ):
            reference_f32 = reference.float()
            candidate_f32 = candidate.float()
            if reference.shape != candidate.shape or not torch.isfinite(candidate_f32).all():
                raise RuntimeError(
                    f"Invalid optimized text embedding for {prompt!r}: "
                    f"reference_shape={tuple(reference.shape)} candidate_shape={tuple(candidate.shape)}"
                )
            parity.append(
                {
                    "prompt": prompt,
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(
                            reference_f32.flatten(), candidate_f32.flatten(), dim=0
                        ).item()
                    ),
                    "relative_l2": float(
                        torch.linalg.vector_norm(candidate_f32 - reference_f32).div(
                            torch.linalg.vector_norm(reference_f32).clamp_min(1e-12)
                        ).item()
                    ),
                    "max_abs": float((candidate_f32 - reference_f32).abs().max().item()),
                }
            )
        text_encoder_validation = {
            "backend": text_encoder_backend,
            "base_model_forward": True,
            "lm_head_skipped": True,
            "original_decoder_layers": original_layer_count,
            "executed_decoder_layers": required_layer_count,
            "nvfp4_linear_count": nvfp4_linear_count,
            "embedding_parity": parity,
            "minimum_cosine": min(item["cosine"] for item in parity),
        }
        print(f"text_encoder_validation={json.dumps(text_encoder_validation, sort_keys=True)}")
        if compile_text_encoder:
            pipe.text_encoder.model = torch.compile(
                decoder,
                mode="max-autotune",
                fullgraph=False,
                dynamic=False,
            )
        del reference_embeddings, candidate_embeddings
        pipe._prompt_cache.clear()
    elif text_encoder_backend == "compressed-tensors-nvfp4":
        from transformers import Qwen3ForCausalLM

        validation_prompts = prompt_list[:3]
        pipe._prompt_cache.clear()
        with torch.inference_mode():
            reference_embeddings = [
                pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)[0].detach().clone()
                for prompt in validation_prompts
            ]
        pipe._prompt_cache.clear()

        old_text_encoder = pipe.text_encoder
        del pipe.text_encoder
        del old_text_encoder
        torch.cuda.empty_cache()
        allocated_before = torch.cuda.memory_allocated()
        print(f"text_encoder_load=start backend={text_encoder_backend} path={nvfp4_text_encoder_dir}")
        pipe.text_encoder = Qwen3ForCausalLM.from_pretrained(
            nvfp4_text_encoder_dir,
            # Keep non-quantized layers and compressed-tensors lifecycle hooks
            # aligned with the checkpoint's BF16 compute dtype.
            torch_dtype=dtype,
            local_files_only=True,
        ).eval().to("cuda")
        allocated_after = torch.cuda.memory_allocated()

        quantization_config = getattr(pipe.text_encoder.config, "quantization_config", None)
        if quantization_config is None:
            raise RuntimeError("NVFP4 text encoder config has no quantization_config")
        if hasattr(quantization_config, "to_dict"):
            quantization_config_dict = quantization_config.to_dict()
        elif isinstance(quantization_config, dict):
            quantization_config_dict = quantization_config
        else:
            raise RuntimeError(
                "Unsupported text encoder quantization config type: "
                f"{type(quantization_config)!r}"
            )
        quant_method = str(quantization_config_dict.get("quant_method", "")).lower().replace("_", "-")
        run_compressed = bool(quantization_config_dict.get("run_compressed", True))
        hf_quantizer = getattr(pipe.text_encoder, "hf_quantizer", None)
        quantizer_name = type(hf_quantizer).__name__ if hf_quantizer is not None else None
        runtime_run_compressed = bool(getattr(hf_quantizer, "run_compressed", False))
        packed_parameter_names = [
            name
            for name, _parameter in pipe.text_encoder.named_parameters()
            if name.endswith(("weight_packed", "weight_scale", "weight_shape"))
        ]
        if (
            "compressed-tensors" not in quant_method
            or not run_compressed
            or not runtime_run_compressed
            or not packed_parameter_names
        ):
            raise RuntimeError(
                "The checkpoint was decompressed by the Transformers compressed-tensors loader; "
                "this is not an NVFP4 inference path and will not be benchmarked. "
                f"quant_method={quant_method!r} run_compressed={run_compressed} "
                f"runtime_run_compressed={runtime_run_compressed} "
                f"packed_parameter_count={len(packed_parameter_names)} "
                f"hf_quantizer={quantizer_name!r}"
            )
        instrumented_modules = [
            name
            for name, module in pipe.text_encoder.named_modules()
            if module.__class__.__module__.startswith("compressed_tensors")
            or hasattr(module, "quantization_scheme")
            or hasattr(module, "weight_packed")
        ]
        if not instrumented_modules:
            raise RuntimeError(
                "NVFP4 text encoder loaded without compressed-tensors instrumentation; "
                "refusing a possible BF16 fallback"
            )

        candidate_embeddings = []
        with torch.inference_mode():
            for prompt in validation_prompts:
                pipe._prompt_cache.clear()
                candidate_embeddings.append(
                    pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)[0].detach().clone()
                )
        parity = []
        for prompt, reference, candidate in zip(
            validation_prompts, reference_embeddings, candidate_embeddings, strict=True
        ):
            if reference.shape != candidate.shape or not torch.isfinite(candidate).all():
                raise RuntimeError(
                    f"Invalid NVFP4 text embedding for {prompt!r}: "
                    f"reference_shape={tuple(reference.shape)} candidate_shape={tuple(candidate.shape)}"
                )
            reference_f32 = reference.float()
            candidate_f32 = candidate.float()
            parity.append(
                {
                    "prompt": prompt,
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(
                            reference_f32.flatten(), candidate_f32.flatten(), dim=0
                        ).item()
                    ),
                    "relative_l2": float(
                        torch.linalg.vector_norm(candidate_f32 - reference_f32).div(
                            torch.linalg.vector_norm(reference_f32).clamp_min(1e-12)
                        ).item()
                    ),
                    "max_abs": float((candidate_f32 - reference_f32).abs().max().item()),
                }
            )
        text_encoder_validation = {
            "backend": text_encoder_backend,
            "path": nvfp4_text_encoder_dir,
            "instrumented_module_count": len(instrumented_modules),
            "hf_quantizer": quantizer_name,
            "quant_method": quant_method,
            "run_compressed": run_compressed,
            "runtime_run_compressed": runtime_run_compressed,
            "packed_parameter_count": len(packed_parameter_names),
            "cuda_memory_mib": (allocated_after - allocated_before) / (1024**2),
            "embedding_parity": parity,
            "minimum_cosine": min(item["cosine"] for item in parity),
        }
        print(f"text_encoder_validation={json.dumps(text_encoder_validation, sort_keys=True)}")
        del reference_embeddings, candidate_embeddings
        pipe._prompt_cache.clear()
    prequant_qkv_fusion_report = getattr(pipe.transformer, "_klein_prequant_qkv_fusion_report", None)
    if backend == "torchao-nvfp4" and prequant_qkv_fusion_report is None:
        raise RuntimeError("TorchAO loader did not produce a QKV fusion report")
    expected_full_compute_gemms = (
        int(prequant_qkv_fusion_report["expected_nvfp4_gemms_per_step"]) * num_inference_steps
        if backend == "torchao-nvfp4"
        else 436
    )
    transformer_compute_steps = 2 if enable_denoiser_step_reuse else num_inference_steps
    expected_executed_nvfp4_gemms = (
        expected_full_compute_gemms // num_inference_steps
    ) * transformer_compute_steps
    expected_attention_invocations = 25 * transformer_compute_steps
    if use_taef2:
        replace_pipeline_vae_with_taef2(pipe, cache_dir=taef2_cache_dir)
        if hasattr(pipe.vae, "taesd") and hasattr(pipe.vae.taesd, "decoder"):
            pipe.vae.taesd.decoder.to(memory_format=torch.channels_last)
    else:
        if hasattr(pipe.vae, "fuse_qkv_projections"):
            pipe.vae.fuse_qkv_projections()
        pipe.vae.to(memory_format=torch.channels_last)

    pipe.vae.to(memory_format=torch.channels_last)

    # Create stable BN constants before compiling/capturing the TAEF2 encoder.
    # Constants created inside a CUDA Graph belong to its replay pool and cannot
    # safely be retained by the pipeline across invocations.
    vae_constant_ref = torch.empty(
        (), device=pipe.vae.bn.running_mean.device, dtype=pipe.vae.dtype
    )
    pipe._get_vae_bn_constants(vae_constant_ref)

    cache_report = None
    if enable_denoiser_step_reuse:
        if enable_cache or enable_full_denoise_compile:
            raise ValueError(
                "denoiser step reuse, Cache-DiT, and whole-loop compilation are separate paths"
            )
        pipe.configure_denoiser_step_reuse((True, True, False, False))
    if enable_cache:
        cache_report = enable_cache_dit(
            pipe,
            num_inference_steps=num_inference_steps,
            steps_mask=cache_steps_mask,
            residual_diff_threshold=cache_residual_diff_threshold,
            single_block_rdt_scale=cache_single_block_rdt_scale,
            require_reuse_candidate=require_cache_reuse_candidate,
            steps_computation_policy=cache_steps_computation_policy,
        )
    if enable_direct_nvfp4_dispatch and backend != "torchao-nvfp4":
        raise ValueError("direct NVFP4 dispatch is only supported by the TorchAO backend")
    if enable_static_transformer_activation_scales and not enable_direct_nvfp4_dispatch:
        raise ValueError("static transformer activation scales require direct NVFP4 dispatch")
    if enable_static_transformer_activation_scales and (enable_cache or enable_denoiser_step_reuse):
        raise ValueError(
            "static activation calibration is an exact four-forward A/B track; "
            "cache and denoiser-step reuse must be disabled"
        )
    if enable_static_transformer_activation_scales and (
        (height, width) != (576, 384) or num_inference_steps != 4
    ):
        raise ValueError(
            "static transformer activation scales are calibrated only for the "
            "576x384 four-step production shape"
        )

    static_activation_scales = None
    static_activation_calibration_report = None
    static_activation_dynamic_reference = None
    if enable_static_transformer_activation_scales:
        calibration_seed = generator_seed + 1000

        def run_static_activation_calibration() -> None:
            for index, calibration_prompt in enumerate(
                STATIC_ACTIVATION_CALIBRATION_PROMPTS
            ):
                pipe._prompt_cache.clear()
                pipe(
                    prompt=calibration_prompt,
                    image=input_image,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator(device="cuda").manual_seed(
                        calibration_seed + index
                    ),
                    output_type="latent",
                )

        print(
            "static_nvfp4_calibration=start "
            f"prompts={len(STATIC_ACTIVATION_CALIBRATION_PROMPTS)} "
            f"steps={num_inference_steps} resolution={height}x{width}"
        )
        static_activation_scales, static_activation_calibration_report = (
            calibrate_torchao_nvfp4_activation_scales(
                pipe.transformer,
                run_static_activation_calibration,
            )
        )
        pipe._prompt_cache.clear()
        parity_image = pipe.image_processor.preprocess(
            input_image,
            height=height,
            width=width,
            resize_mode="crop",
        )
        parity_latent_height = 2 * (height // (pipe.vae_scale_factor * 2))
        parity_latent_width = 2 * (width // (pipe.vae_scale_factor * 2))
        parity_latents = torch.randn(
            (
                1,
                pipe.transformer.config.in_channels,
                parity_latent_height // 2,
                parity_latent_width // 2,
            ),
            generator=torch.Generator(device="cuda").manual_seed(generator_seed),
            device="cuda",
            dtype=dtype,
        )
        with torch.inference_mode():
            static_activation_dynamic_reference = pipe(
                prompt=STATIC_ACTIVATION_PARITY_PROMPT,
                image=parity_image,
                latents=parity_latents,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(generator_seed),
                output_type="latent",
            ).images.detach().clone()
        del parity_image, parity_latents
        pipe._prompt_cache.clear()
        print(
            "static_nvfp4_calibration="
            + json.dumps(static_activation_calibration_report, sort_keys=True)
        )
    requested_nvfp4_gemm_backend = nvfp4_gemm_backend
    sm120_nvfp4_extension = None
    sm120_nvfp4_setup_error = None
    if nvfp4_gemm_backend == "cutlass-sm120":
        try:
            sm120_nvfp4_extension = _build_sm120_nvfp4_extension()
        except Exception as exc:
            sm120_nvfp4_setup_error = f"{type(exc).__name__}: {exc}"
            nvfp4_gemm_backend = "torch-scaled-mm"
            print(
                "sm120_nvfp4_extension=fallback backend=torch-scaled-mm "
                f"reason={sm120_nvfp4_setup_error}"
            )
    klein_cuda_ops_loaded = _build_klein_cuda_extension(required=True) if enable_klein_cuda_ops else False
    attention_backend = prepare_transformer_for_speed(
        pipe,
        backend=attention_backend_request,
        fuse_qkv=fuse_qkv_before_quantization,
        patch_klein_ops=klein_cuda_ops_loaded,
        direct_nvfp4_dispatch=enable_direct_nvfp4_dispatch,
        fused_qkv_packing=enable_fused_qkv_packing,
        nvfp4_gemm_backend=nvfp4_gemm_backend,
        static_activation_scales=static_activation_scales,
        allow_approximate_attention=(
            optimization_profile
            in {"one-shot-fast", "one-shot-aggressive", "one-shot-80", "one-shot-80-quality"}
        ),
    )
    flashinfer_attention_selected = (attention_backend or "").startswith("flashinfer-")
    selected_attention_is_approximate = attention_backend in {
        "flashinfer-single-prefill-fp16-reduction",
        "flashinfer-nvfp4-sm120",
    }
    direct_nvfp4_report = getattr(pipe.transformer, "_klein_nvfp4_dispatch_report", None)
    attention_selection_report = getattr(pipe.transformer, "_klein_attention_selection_report", None)
    if enable_direct_nvfp4_dispatch and not direct_nvfp4_report:
        raise RuntimeError("direct NVFP4 dispatch was requested but produced no patch report")
    if direct_nvfp4_report:
        print(
            "direct_nvfp4_selection="
            + json.dumps(
                {
                    "selected_backend_counts": direct_nvfp4_report.get("selected_backend_counts"),
                    "shape_backend_choices": direct_nvfp4_report.get("shape_backend_choices"),
                    "shape_kernel_variants": direct_nvfp4_report.get("shape_kernel_variants"),
                    "native_compile_probe": direct_nvfp4_report.get("native_compile_probe"),
                    "static_activation_scale_count": direct_nvfp4_report.get(
                        "static_activation_scale_count"
                    ),
                },
                sort_keys=True,
            )
        )
    if optimization_profile in {
        "one-shot-exact",
        "one-shot-fast",
        "one-shot-aggressive",
        "one-shot-80",
        "one-shot-80-quality",
    }:
        qkv_fusion_active = bool(prequant_qkv_fusion_report["enabled"])
        expected_patched = 89 if qkv_fusion_active else 109
        expected_row_counts = (
            {1: 6, 512: 21, 1728: 22, 2240: 40}
            if qkv_fusion_active
            else {1: 6, 512: 31, 1728: 32, 2240: 40}
        )
        observed_row_counts = {
            int(rows): int(count)
            for rows, count in direct_nvfp4_report.get("production_row_counts", {}).items()
        }
        selected_backend_counts = direct_nvfp4_report.get("selected_backend_counts", {})
        observed_selected = sum(int(count) for count in selected_backend_counts.values())
        requires_native_shape_validation = nvfp4_gemm_backend == "cutlass-sm120"
        observed_static_scales = int(
            direct_nvfp4_report.get("static_activation_scale_count", 0)
        )
        if (
            direct_nvfp4_report.get("patched") != expected_patched
            or observed_selected != expected_patched
            or (requires_native_shape_validation and observed_row_counts != expected_row_counts)
            or (
                enable_static_transformer_activation_scales
                and observed_static_scales != expected_patched
            )
        ):
            raise RuntimeError(
                "one-shot native dispatch topology mismatch: "
                f"patched={direct_nvfp4_report.get('patched')} "
                f"expected_patched={expected_patched} "
                f"static_scales={observed_static_scales} "
                f"selected_backends={selected_backend_counts} "
                f"expected_rows={expected_row_counts} observed_rows={observed_row_counts}"
            )
        print(
            "one_shot_native_topology=validated "
            f"linears={expected_patched} qkv_fusion={qkv_fusion_active} "
            f"backend={nvfp4_gemm_backend} "
            f"production_row_counts={json.dumps(observed_row_counts, sort_keys=True)}"
        )
    fused_qkv_packing_report = getattr(pipe.transformer, "_klein_fused_qkv_packing_report", None)
    if enable_fused_qkv_packing and not fused_qkv_packing_report:
        raise RuntimeError("fused QKV packing was requested but produced no patch report")
    klein_cuda_ops_patched = bool(getattr(pipe.transformer, "_klein_cuda_ops_patched", False))
    if enable_klein_cuda_ops and not klein_cuda_ops_patched:
        raise RuntimeError("Klein CUDA operators were requested but no transformer blocks were patched")

    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.epilogue_fusion = False

    if enable_full_denoise_compile and enable_cache:
        raise ValueError("whole-loop compile and Cache-DiT must be benchmarked separately")
    compile_transformer = (
        enable_torch_compile
        and backend != "modelopt-nvfp4"
        and not enable_full_denoise_compile
    )
    transformer_compile_mode = "max-autotune-no-cudagraphs" if enable_cache else "max-autotune"
    if enable_full_denoise_compile:
        enable_compiled_denoise_loop(
            pipe,
            num_inference_steps=num_inference_steps,
            mode="max-autotune",
        )
    elif compile_transformer:
        if enable_cache:
            pipe._cache_dit_mod.set_compile_configs()
            print("cache_dit_compile_configs=applied")
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode=transformer_compile_mode,
            fullgraph=False,
            dynamic=False,
        )
    elif backend == "modelopt-nvfp4":
        print("modelopt_nvfp4_transformer_compile=disabled (ModelOpt dynamic callbacks are not Dynamo-safe)")
    else:
        print("transformer_compile=disabled")

    def _vae_encode_fn(image: torch.Tensor, generator: torch.Generator):
        return pipe._encode_vae_image(image=image, generator=generator)

    pipe._vae_encode_fn = (
        torch.compile(
            _vae_encode_fn,
            # The encoder output is consumed by the separately captured denoiser.
            # CUDA Graph Trees cannot safely transfer graph-owned output storage
            # between those compiled regions, so retain Inductor/autotuning but
            # leave graph capture to the four-step denoiser.
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
            dynamic=False,
        )
        if enable_torch_compile
        else _vae_encode_fn
    )

    def _vae_decode_fn(latents: torch.Tensor):
        return pipe.vae.decode(latents, return_dict=False)[0]

    pipe._vae_decode_fn = (
        torch.compile(
            _vae_decode_fn,
            mode="max-autotune",
            fullgraph=False,
            dynamic=False,
        )
        if enable_torch_compile
        else _vae_decode_fn
    )
    if not enable_torch_compile:
        print("taef2_compile=disabled encode=eager decode=eager")
    pipe._nsight_nvtx_enabled = nsight_capture

    latent_height = 2 * (height // (pipe.vae_scale_factor * 2))
    latent_width = 2 * (width // (pipe.vae_scale_factor * 2))
    prepared_image = pipe.image_processor.preprocess(
        input_image,
        height=height,
        width=width,
        resize_mode="crop",
    )
    if prepared_image.device.type == "cpu" and not prepared_image.is_pinned():
        prepared_image = prepared_image.pin_memory()

    last_prompt_events = None
    last_vae_events = None
    last_preparation_events = None
    prompt_stream = torch.cuda.Stream() if overlap_preparation else None
    vae_stream = torch.cuda.Stream() if overlap_preparation else None
    preparation_start_event = torch.cuda.Event(enable_timing=True) if overlap_preparation else None
    prompt_ready_event = torch.cuda.Event() if overlap_preparation else None
    image_ready_event = torch.cuda.Event() if overlap_preparation else None
    prompt_timing_start = torch.cuda.Event(enable_timing=True)
    prompt_timing_end = torch.cuda.Event(enable_timing=True)
    vae_timing_start = torch.cuda.Event(enable_timing=True) if overlap_preparation else None
    vae_timing_end = torch.cuda.Event(enable_timing=True) if overlap_preparation else None
    preparation_timing_end = torch.cuda.Event(enable_timing=True) if overlap_preparation else None

    def call_pipeline(
        prompt: str,
        *,
        callback_on_step_end=None,
        record_prompt_timing: bool = False,
        output_type_override: str | None = None,
    ):
        nonlocal last_preparation_events, last_prompt_events, last_vae_events
        prompt_start = prompt_timing_start if record_prompt_timing else None
        prompt_end = prompt_timing_end if record_prompt_timing else None
        vae_start = vae_timing_start if record_prompt_timing else None
        vae_end = vae_timing_end if record_prompt_timing else None
        image_latents = None
        image_latent_ids = None
        if overlap_preparation:
            current_stream = torch.cuda.current_stream()
            preparation_start = preparation_start_event
            prompt_ready = prompt_ready_event
            image_ready = image_ready_event
            preparation_start.record(current_stream)
            prompt_stream.wait_event(preparation_start)
            vae_stream.wait_event(preparation_start)
            with torch.cuda.stream(prompt_stream):
                with torch.cuda.nvtx.range("prompt_encode") if nsight_capture else nullcontext():
                    if prompt_start is not None:
                        prompt_start.record(prompt_stream)
                    prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)
                    prompt_embeds = prompt_embeds.to(dtype=dtype)
                    if prompt_end is not None:
                        prompt_end.record(prompt_stream)
                    prompt_ready.record(prompt_stream)
            with torch.cuda.stream(vae_stream):
                with torch.cuda.nvtx.range("taef2_encode") if nsight_capture else nullcontext():
                    if vae_start is not None:
                        vae_start.record(vae_stream)
                    image_latents, image_latent_ids = pipe.prepare_image_latents(
                        images=[prepared_image],
                        batch_size=1,
                        generator=torch.Generator(device="cuda").manual_seed(generator_seed),
                        device=pipe._execution_device,
                        dtype=pipe.vae.dtype,
                        non_blocking_h2d=True,
                    )
                    if vae_end is not None:
                        vae_end.record(vae_stream)
                    image_ready.record(vae_stream)
            current_stream.wait_event(prompt_ready)
            current_stream.wait_event(image_ready)
            if record_prompt_timing:
                preparation_done = preparation_timing_end
                preparation_done.record(current_stream)
                last_vae_events = (vae_start, vae_end)
                last_preparation_events = (preparation_start, preparation_done)
        else:
            if prompt_start is not None:
                prompt_start.record()
            prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)
            prompt_embeds = prompt_embeds.to(dtype=dtype)
            if prompt_end is not None:
                prompt_end.record()
        if prompt_end is not None:
            last_prompt_events = (prompt_start, prompt_end)
        with torch.cuda.nvtx.range("noise_setup") if nsight_capture else nullcontext():
            fixed_latents = torch.randn(
                (1, pipe.transformer.config.in_channels, latent_height // 2, latent_width // 2),
                generator=torch.Generator(device="cuda").manual_seed(generator_seed),
                device="cuda",
                dtype=dtype,
            )
        with torch.cuda.nvtx.range("pipeline_denoise_decode") if nsight_capture else nullcontext():
            return pipe(
                prompt=None,
                prompt_embeds=prompt_embeds,
                image=None if overlap_preparation else prepared_image,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                latents=fixed_latents,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(generator_seed),
                output_type=output_type if output_type_override is None else output_type_override,
                callback_on_step_end=callback_on_step_end,
            ).images

    static_activation_parity = None
    if static_activation_dynamic_reference is not None:
        pipe._prompt_cache.clear()
        with torch.inference_mode():
            static_candidate = call_pipeline(
                STATIC_ACTIVATION_PARITY_PROMPT,
                output_type_override="latent",
            )
        torch.cuda.synchronize()
        dynamic_f32 = static_activation_dynamic_reference.float()
        static_f32 = static_candidate.float()
        difference = static_f32 - dynamic_f32
        static_activation_parity = {
            "prompt": STATIC_ACTIVATION_PARITY_PROMPT,
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    dynamic_f32.flatten(), static_f32.flatten(), dim=0
                ).item()
            ),
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                .div(torch.linalg.vector_norm(dynamic_f32).clamp_min(1e-12))
                .item()
            ),
            "max_abs": float(difference.abs().max().item()),
            "finite": bool(torch.isfinite(static_f32).all().item()),
        }
        print(
            "static_nvfp4_parity="
            + json.dumps(static_activation_parity, sort_keys=True)
        )
        if not static_activation_parity["finite"]:
            raise RuntimeError(
                "calibrated static NVFP4 transformer produced non-finite latents: "
                f"{static_activation_parity}"
            )
        del static_activation_dynamic_reference, static_candidate
        pipe._prompt_cache.clear()

    cache_validation = None

    validated_steps: list[int] = []

    def is_sm120_blockscaled_gemm_kernel(event: dict) -> bool:
        if event.get("cat") != "kernel":
            return False
        name = str(event.get("name", ""))
        is_cutlass_device_kernel = (
            "cutlass::device_kernel" in name or "_ZN7cutlass13device_kernel" in name
        )
        return is_cutlass_device_kernel and "Sm120" in name and "BlockScaled" in name

    def record_step(_pipe, step, _timestep, callback_kwargs):
        validated_steps.append(int(step))
        return callback_kwargs

    with torch.inference_mode():
        reference_output = call_pipeline(WARMUP_PROMPT, callback_on_step_end=record_step)
    torch.cuda.synchronize()
    if validated_steps != list(range(num_inference_steps)):
        raise RuntimeError(f"Expected {num_inference_steps} denoising steps, observed {validated_steps}")

    denoise_loop_validation = None
    if enable_full_denoise_compile:
        with torch.inference_mode():
            compiled_output = call_pipeline(WARMUP_PROMPT)
        torch.cuda.synchronize()
        reference_tensor = reference_output[0] if isinstance(reference_output, list) else reference_output
        compiled_tensor = compiled_output[0] if isinstance(compiled_output, list) else compiled_output
        reference_f32 = reference_tensor.float()
        compiled_f32 = compiled_tensor.float()
        max_abs = float((reference_f32 - compiled_f32).abs().max().item())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                reference_f32.flatten(),
                compiled_f32.flatten(),
                dim=0,
            ).item()
        )
        denoise_loop_validation = {
            "max_abs": max_abs,
            "cosine": cosine,
            "accepted": True,
            "fallback": None,
        }
        print(f"denoise_loop_validation={json.dumps(denoise_loop_validation, sort_keys=True)}")
        if not torch.allclose(reference_f32, compiled_f32, rtol=2e-2, atol=2e-2):
            denoise_loop_validation["accepted"] = False
            denoise_loop_validation["fallback"] = "compiled-transformer-ordinary-loop"
            print(
                "compiled_denoise_loop=rejected reason=parity "
                f"validation={json.dumps(denoise_loop_validation, sort_keys=True)}"
            )
            pipe._compiled_denoise_loop = None
            pipe._compiled_denoise_loop_steps = None
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode="max-autotune",
                fullgraph=False,
                dynamic=False,
            )

    warmup_times: list[float] = []
    for i in range(warmup_runs):
        # The measured contract is unique prompts, so warm the same execution
        # path rather than repeatedly hitting the prompt-embedding cache.
        pipe._prompt_cache.clear()
        warmup_prompt = f"uncached compilation warmup prompt {i + 1}"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            call_pipeline(warmup_prompt)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        warmup_times.append(dt_ms)
        print(f"Warmup {i + 1}: {dt_ms:.1f} ms")

    print(f"Warmup avg: {sum(warmup_times) / len(warmup_times):.1f} ms")

    if enable_cache:
        import torch.profiler

        # Profile only after compile warmup. The prompt is cached solely for this
        # diagnostic trace; all measured requests below still use fresh prompts.
        pipe._prompt_cache.clear()
        with torch.inference_mode():
            pipe.encode_prompt(prompt=WARMUP_PROMPT, device=pipe._execution_device)
        trace_path = Path("/tmp/klein4b_cache_validation.json")
        with torch.inference_mode(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as cache_profiler:
            call_pipeline(WARMUP_PROMPT)
        cache_profiler.export_chrome_trace(str(trace_path))
        trace_events = json.loads(trace_path.read_text()).get("traceEvents", [])
        gpu_kernel_events = [event for event in trace_events if event.get("cat") == "kernel"]
        gemm_events = [
            event
            for event in gpu_kernel_events
            if "cutlass3x_sm120_bstensorop" in str(event.get("name", ""))
            or is_sm120_blockscaled_gemm_kernel(event)
        ]
        if flashinfer_attention_selected:
            attention_events = [
                event
                for event in gpu_kernel_events
                if any(
                    fragment in str(event.get("name", "")).lower()
                    for fragment in (
                        "flashinfer",
                        "single_prefill",
                        "singleprefill",
                        "prefill",
                        "nvfp4_attention",
                    )
                )
            ]
        else:
            attention_events = [
                event for event in gpu_kernel_events if "flash_fwd" in str(event.get("name", ""))
            ]
        gemm_count = len(gemm_events)
        gemm_reduction = 1.0 - gemm_count / expected_full_compute_gemms
        forced_compute_steps = int((cache_report or {}).get("forced_compute_steps", num_inference_steps))
        reuse_candidate_steps = int((cache_report or {}).get("reuse_candidate_steps", 0))
        flashinfer_attention_op_events = [
            event
            for event in trace_events
            if "klein::flashinfer_" in str(event.get("name", ""))
        ]
        attention_invocation_count = (
            len(flashinfer_attention_op_events)
            if flashinfer_attention_selected
            else len(attention_events)
        )
        expected_static_mask_gemms = (
            forced_compute_steps * (expected_full_compute_gemms // num_inference_steps)
            + reuse_candidate_steps * 9
            if cache_steps_computation_policy == "static"
            else None
        )
        cache_validation = {
            "nvfp4_gemm_count": gemm_count,
            "nvfp4_gemm_ms": sum(float(event.get("dur", 0.0)) for event in gemm_events) / 1000.0,
            "baseline_full_compute_gemms": expected_full_compute_gemms,
            "gemms_removed": expected_full_compute_gemms - gemm_count,
            "gemm_reduction_fraction": gemm_reduction,
            "attention_kernel_count": len(attention_events),
            "attention_invocation_count": attention_invocation_count,
            "attention_kernel_ms": sum(float(event.get("dur", 0.0)) for event in attention_events)
            / 1000.0,
            "baseline_attention_invocations": 25 * num_inference_steps,
            "expected_static_mask_gemms": expected_static_mask_gemms,
            "expected_static_mask_attention_invocations": (
                forced_compute_steps * 25
                if cache_steps_computation_policy == "static"
                else None
            ),
            "steps_mask": cache_steps_mask,
            "steps_computation_policy": cache_steps_computation_policy,
        }
        print(f"cache_validation={json.dumps(cache_validation, sort_keys=True)}")
        minimum_reduction = {
            "one-shot-fast": 0.20,
            "one-shot-aggressive": 0.44,
        }.get(optimization_profile, 0.0)
        expected_static_attention = cache_validation[
            "expected_static_mask_attention_invocations"
        ]
        if (
            gemm_count >= expected_full_compute_gemms
            or gemm_reduction < minimum_reduction
            or (
                expected_static_mask_gemms is not None
                and gemm_count != expected_static_mask_gemms
            )
            or (
                expected_static_attention is not None
                and attention_invocation_count != expected_static_attention
            )
        ):
            raise RuntimeError(
                "Cache-DiT did not execute the required static work pattern: "
                f"minimum_reduction={minimum_reduction} validation={cache_validation}"
            )

    direct_dispatch_validation = None
    if enable_direct_nvfp4_dispatch and enable_internal_profiler_validation:
        import torch.profiler

        # This profiler validates transformer dispatch and exact GEMM counts.
        # Keep text encoding outside the trace; the primary benchmark below
        # still clears this cache and measures 20 fresh unique prompts.
        pipe._prompt_cache.clear()
        with torch.inference_mode():
            pipe.encode_prompt(prompt=WARMUP_PROMPT, device=pipe._execution_device)
        trace_path = Path("/tmp/klein4b_direct_nvfp4_validation.json")
        with torch.inference_mode(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as direct_profiler:
            call_pipeline(WARMUP_PROMPT)
        direct_profiler.export_chrome_trace(str(trace_path))
        trace_events = json.loads(trace_path.read_text()).get("traceEvents", [])

        def matching_events(fragment: str) -> list[dict]:
            return [event for event in trace_events if fragment in str(event.get("name", ""))]

        native_gemm_op_events = matching_events("klein_sm120::nvfp4_gemm_out")
        if nvfp4_gemm_backend == "cutlass-sm120":
            native_external_ids = {
                event.get("args", {}).get("External id")
                for event in native_gemm_op_events
                if event.get("args", {}).get("External id") is not None
            }
            native_gemm_kernel_events = [
                event
                for event in trace_events
                if is_sm120_blockscaled_gemm_kernel(event)
                or (
                    event.get("cat") == "kernel"
                    and native_external_ids
                    and event.get("args", {}).get("External id") in native_external_ids
                )
            ]
            torch_gemm_events = [
                event
                for event in matching_events("cutlass3x_sm120_bstensorop")
                if event.get("args", {}).get("External id") not in native_external_ids
            ]
            gemm_events = native_gemm_kernel_events + torch_gemm_events
            observed_gemm_count = len(native_gemm_kernel_events) + len(torch_gemm_events)
        else:
            native_gemm_kernel_events = []
            gemm_events = matching_events("cutlass3x_sm120_bstensorop")
            observed_gemm_count = len(gemm_events)
        subclass_events = matching_events("PythonSubclass")
        nvfp4_linear_events = matching_events("nvfp4_linear")
        torch_dispatch_events = matching_events("_dispatch__torch_dispatch__")
        quantize_events = matching_events("mslk_quantize_nvfp4")
        fused_qkv_pack_events = matching_events("_fused_joint_qkv_rmsnorm_rope_pack_kernel")
        fused_single_qkv_pack_events = matching_events("_fused_single_qkv_rmsnorm_rope_pack_kernel")
        fused_post_attention_events = matching_events("_fused_attention_swiglu_pack_kernel")
        rope_build_events = matching_events("get_1d_rotary_pos_embed")
        graph_launch_events = matching_events("cudaGraphLaunch")
        allocation_events = matching_events("cudaMalloc") + matching_events("cudaFree")
        synchronization_events = (
            matching_events("cudaDeviceSynchronize")
            + matching_events("cudaStreamSynchronize")
            + matching_events("cudaEventSynchronize")
        )
        gpu_kernel_events = [event for event in trace_events if event.get("cat") == "kernel"]
        kernel_totals: dict[str, dict[str, float | int]] = {}
        for event in gpu_kernel_events:
            name = str(event.get("name", ""))
            aggregate = kernel_totals.setdefault(name, {"count": 0, "ms": 0.0})
            aggregate["count"] = int(aggregate["count"]) + 1
            aggregate["ms"] = float(aggregate["ms"]) + float(event.get("dur", 0.0)) / 1000.0
        top_gpu_kernels = [
            {
                "name": name if len(name) <= 240 else f"{name[:237]}...",
                "count": values["count"],
                "ms": values["ms"],
            }
            for name, values in sorted(
                kernel_totals.items(),
                key=lambda item: float(item[1]["ms"]),
                reverse=True,
            )[:20]
        ]
        selected_backend_counts = (direct_nvfp4_report or {}).get("selected_backend_counts", {})
        expected_native_gemm_count = (
            int(selected_backend_counts.get("cutlass-sm120", 0))
            * transformer_compute_steps
        )
        expected_torch_gemm_count = (
            int(selected_backend_counts.get("torch-scaled-mm", 0))
            * transformer_compute_steps
        )
        if attention_backend == "cudnn-sdpa":
            attention_events = [
                event
                for event in gpu_kernel_events
                if any(
                    fragment in str(event.get("name", "")).lower()
                    for fragment in ("fmha", "attention", "cudnn")
                )
            ]
        elif flashinfer_attention_selected:
            attention_events = [
                event
                for event in gpu_kernel_events
                if any(
                    fragment in str(event.get("name", "")).lower()
                    for fragment in (
                        "flashinfer",
                        "single_prefill",
                        "singleprefill",
                        "prefill",
                        "nvfp4_attention",
                    )
                )
            ]
        else:
            attention_events = matching_events("flash_fwd")
        dynamo_lookup_events = matching_events("TorchDynamo Cache Lookup")
        copy_events = [event for event in trace_events if str(event.get("name", "")) == "aten::copy_"]
        to_copy_events = [event for event in trace_events if str(event.get("name", "")) == "aten::_to_copy"]
        cuda_copy_events = [
            event
            for event in trace_events
            if event.get("cat") == "gpu_memcpy"
            or (
                event.get("cat") == "kernel"
                and any(
                    fragment in str(event.get("name", "")).lower()
                    for fragment in ("copy_kernel", "batchedcopy", "memcpy")
                )
            )
        ]
        direct_dispatch_validation = {
            "nvfp4_gemm_count": observed_gemm_count,
            "nvfp4_gemm_ms": sum(float(event.get("dur", 0.0)) for event in gemm_events) / 1000.0,
            "native_sm120_gemm_op_count": len(native_gemm_op_events),
            "native_sm120_gemm_kernel_count": len(native_gemm_kernel_events),
            "torch_scaled_mm_gemm_count": (
                len(torch_gemm_events) if nvfp4_gemm_backend == "cutlass-sm120" else observed_gemm_count
            ),
            "expected_native_sm120_gemm_count": expected_native_gemm_count,
            "expected_torch_scaled_mm_gemm_count": expected_torch_gemm_count,
            "python_subclass_count": len(subclass_events),
            "python_subclass_ms": sum(float(event.get("dur", 0.0)) for event in subclass_events) / 1000.0,
            "nvfp4_linear_dispatch_count": len(nvfp4_linear_events),
            "torch_dispatch_count": len(torch_dispatch_events),
            "mslk_quantize_count": len(quantize_events),
            "mslk_quantize_ms": sum(float(event.get("dur", 0.0)) for event in quantize_events) / 1000.0,
            "fused_qkv_pack_count": len(fused_qkv_pack_events),
            "fused_qkv_pack_ms": sum(float(event.get("dur", 0.0)) for event in fused_qkv_pack_events) / 1000.0,
            "fused_single_qkv_pack_count": len(fused_single_qkv_pack_events),
            "fused_single_qkv_pack_ms": sum(
                float(event.get("dur", 0.0)) for event in fused_single_qkv_pack_events
            )
            / 1000.0,
            "fused_post_attention_count": len(fused_post_attention_events),
            "fused_post_attention_ms": sum(
                float(event.get("dur", 0.0)) for event in fused_post_attention_events
            )
            / 1000.0,
            "attention_count": len(attention_events),
            "attention_ms": sum(float(event.get("dur", 0.0)) for event in attention_events)
            / 1000.0,
            "attention_backend": attention_backend,
            "expected_attention_invocations": expected_attention_invocations,
            "gpu_kernel_count": len(gpu_kernel_events),
            "gpu_kernel_ms": sum(float(event.get("dur", 0.0)) for event in gpu_kernel_events)
            / 1000.0,
            "rope_build_count": len(rope_build_events),
            "rope_build_ms": sum(float(event.get("dur", 0.0)) for event in rope_build_events) / 1000.0,
            "cuda_graph_launch_count": len(graph_launch_events),
            "dynamo_lookup_count": len(dynamo_lookup_events),
            "dynamo_lookup_ms": sum(float(event.get("dur", 0.0)) for event in dynamo_lookup_events) / 1000.0,
            "copy_cpu_op_count": len(copy_events),
            "copy_cpu_span_ms": sum(float(event.get("dur", 0.0)) for event in copy_events) / 1000.0,
            "to_copy_cpu_op_count": len(to_copy_events),
            "to_copy_cpu_span_ms": sum(float(event.get("dur", 0.0)) for event in to_copy_events) / 1000.0,
            "cuda_copy_event_count": len(cuda_copy_events),
            "cuda_copy_event_ms": sum(float(event.get("dur", 0.0)) for event in cuda_copy_events)
            / 1000.0,
            "hot_request_allocation_count": len(allocation_events),
            "hot_request_synchronization_count": len(synchronization_events),
            "top_gpu_kernels": top_gpu_kernels,
        }
        print(f"direct_nvfp4_validation={json.dumps(direct_dispatch_validation, sort_keys=True)}")
        if nvfp4_linear_events or torch_dispatch_events:
            raise RuntimeError(
                "direct NVFP4 dispatch validation still observed TorchAO Tensor-subclass dispatch: "
                f"{direct_dispatch_validation}"
            )
        if (
            nvfp4_gemm_backend in {"torch-scaled-mm", "cutlass-sm120"}
            and not enable_cache
            and (
                observed_gemm_count != expected_executed_nvfp4_gemms
                or len(native_gemm_kernel_events) != expected_native_gemm_count
                or (
                    nvfp4_gemm_backend == "cutlass-sm120"
                    and len(torch_gemm_events) != expected_torch_gemm_count
                )
            )
        ):
            raise RuntimeError(
                "direct NVFP4 dispatch observed an unexpected full-compute GEMM count: "
                f"expected_total={expected_executed_nvfp4_gemms}, observed_total={observed_gemm_count}, "
                f"expected_native={expected_native_gemm_count}, "
                f"observed_native={len(native_gemm_kernel_events)}, "
                f"expected_torch={expected_torch_gemm_count}, "
                f"observed_torch={len(torch_gemm_events)} "
                f"top_gpu_kernels={json.dumps(top_gpu_kernels, sort_keys=True)}"
            )
        if rope_build_events:
            raise RuntimeError(
                "resident RoPE validation observed rotary embedding construction in the warmed request: "
                f"{direct_dispatch_validation}"
            )
        if (
            enable_fused_qkv_packing
            and not enable_cache
            and len(fused_qkv_pack_events) != 5 * transformer_compute_steps
        ):
            raise RuntimeError(
                "fused QKV packing did not execute once per double block and denoising step: "
                f"{direct_dispatch_validation}"
            )
        if (
            enable_fused_qkv_packing
            and not enable_cache
            and len(fused_single_qkv_pack_events) != 20 * transformer_compute_steps
        ):
            raise RuntimeError(
                "fused single-block QKV packing did not execute once per block and denoising step: "
                f"{direct_dispatch_validation}"
            )
        if (
            enable_fused_qkv_packing
            and not enable_cache
            and len(fused_post_attention_events) != 20 * transformer_compute_steps
        ):
            raise RuntimeError(
                "fused post-attention packing did not execute once per single block and step: "
                f"{direct_dispatch_validation}"
            )
        if allocation_events:
            raise RuntimeError(
                "warmed optimized request performed CUDA allocation/free operations: "
                f"{direct_dispatch_validation}"
            )
        if (
            not enable_cache
            and attention_backend in {"native", "_flash_3"}
            and len(attention_events) != expected_attention_invocations
        ):
            raise RuntimeError(
                "exact attention validation observed an unexpected kernel count: "
                f"backend={attention_backend} expected={expected_attention_invocations} "
                f"observed={len(attention_events)} validation={direct_dispatch_validation}"
            )
        if flashinfer_attention_selected and not attention_events:
            raise RuntimeError(
                "FlashInfer attention was selected but no FlashInfer prefill kernels were observed: "
                f"{direct_dispatch_validation}"
            )
        native_selected_count = int(
            (direct_nvfp4_report or {}).get("selected_backend_counts", {}).get("cutlass-sm120", 0)
        )
        if (
            enable_torch_compile
            and native_selected_count
            and not enable_cache
            and not graph_launch_events
        ):
            raise RuntimeError(
                "native SM120 dispatch disabled CUDA Graph replay; rejecting the backend: "
                f"selected_native_linears={native_selected_count} "
                f"validation={direct_dispatch_validation}"
            )
        if (
            enable_torch_compile
            and flashinfer_attention_selected
            and not enable_cache
            and len(graph_launch_events) < 5
        ):
            raise RuntimeError(
                "FlashInfer attention did not preserve complete warmed graph replay; rejecting it: "
                f"cuda_graph_launch_count={len(graph_launch_events)} "
                f"validation={direct_dispatch_validation}"
            )

    wall_times: list[float] = []
    cuda_times: list[float] = []
    prompt_cuda_times: list[float] = []
    vae_encode_cuda_times: list[float] = []
    overlapped_preparation_cuda_times: list[float] = []
    measured_images = []
    first_output = None
    if measured_runs != len(prompt_list):
        print(
            f"measured_runs={measured_runs} ignored; benchmark contract uses "
            f"all {len(prompt_list)} unique prompts exactly once"
        )
    pipe._prompt_cache.clear()
    unique_prompt_count = 1 if nsight_capture else len(prompt_list)
    for i in range(unique_prompt_count):
        prompt = prompt_list[i]
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if nsight_capture:
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("klein_e2e_exact")
        try:
            start_event.record()
            with torch.inference_mode():
                output = call_pipeline(prompt, record_prompt_timing=True)
            end_event.record()
            end_event.synchronize()
        finally:
            if nsight_capture:
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        cuda_ms = start_event.elapsed_time(end_event)
        wall_times.append(wall_ms)
        cuda_times.append(cuda_ms)
        if last_prompt_events is None:
            raise RuntimeError("Prompt timing events were not recorded")
        prompt_cuda_times.append(last_prompt_events[0].elapsed_time(last_prompt_events[1]))
        if overlap_preparation:
            if last_vae_events is None or last_preparation_events is None:
                raise RuntimeError("Overlapped preparation timing events were not recorded")
            vae_encode_cuda_times.append(last_vae_events[0].elapsed_time(last_vae_events[1]))
            overlapped_preparation_cuda_times.append(
                last_preparation_events[0].elapsed_time(last_preparation_events[1])
            )
        if first_output is None:
            first_output = output
        if output_type == "pil":
            measured_images.append(output[0])
        print(f"[{i + 1:02d}/{unique_prompt_count:02d}] wall={wall_ms:.3f} ms cuda={cuda_ms:.3f} ms | {prompt}")

    def percentile(values: list[float], p: float) -> float:
        ordered = sorted(values)
        index = (len(ordered) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        fraction = index - lo
        return ordered[lo] * (1 - fraction) + ordered[hi] * fraction

    result = {
        "backend": backend,
        "optimization_profile": optimization_profile,
        "effective_optimization_config": {
            "enable_cache": enable_cache,
            "enable_klein_cuda_ops": enable_klein_cuda_ops,
            "enable_direct_nvfp4_dispatch": enable_direct_nvfp4_dispatch,
            "enable_static_transformer_activation_scales": (
                enable_static_transformer_activation_scales
            ),
            "enable_full_denoise_compile": enable_full_denoise_compile,
            "enable_denoiser_step_reuse": enable_denoiser_step_reuse,
            "fuse_qkv_before_quantization": fuse_qkv_before_quantization,
            "enable_fused_qkv_packing": enable_fused_qkv_packing,
            "nvfp4_gemm_backend": nvfp4_gemm_backend,
            "attention_backend_request": attention_backend_request,
            "allow_approximate_attention": optimization_profile
            in {"one-shot-fast", "one-shot-aggressive", "one-shot-80", "one-shot-80-quality"},
            "cache_steps_mask": cache_steps_mask,
            "cache_steps_computation_policy": cache_steps_computation_policy,
            "compile_text_encoder": compile_text_encoder,
            "enable_torch_compile": enable_torch_compile,
            "nsight_capture": nsight_capture,
            "enable_internal_profiler_validation": enable_internal_profiler_validation,
            "overlap_preparation": overlap_preparation,
            "output_type": output_type,
        },
        "text_encoder_backend": text_encoder_backend,
        "text_encoder_compile": bool(
            compile_text_encoder and text_encoder_backend in {"bf16", "torchao-nvfp4"}
        ),
        "text_encoder_validation": text_encoder_validation,
        "nvfp4_gemm_backend": nvfp4_gemm_backend,
        "requested_nvfp4_gemm_backend": requested_nvfp4_gemm_backend,
        "runtime_versions": {
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
        },
        "modelopt_state_path": modelopt_state_path if backend == "modelopt-nvfp4" else None,
        "transformer_compile": compile_transformer,
        "transformer_compile_mode": transformer_compile_mode if compile_transformer else None,
        "full_denoise_compile": enable_full_denoise_compile,
        "full_denoise_compile_active": bool(
            enable_full_denoise_compile
            and denoise_loop_validation
            and denoise_loop_validation.get("accepted")
        ),
        "denoise_loop_validation": denoise_loop_validation,
        "vae": "taef2" if use_taef2 else "original",
        "resolution": [height, width],
        "num_inference_steps": num_inference_steps,
        "validated_denoising_steps": validated_steps,
        "execution_contract": {
            "scheduler_steps": num_inference_steps,
            "full_transformer_steps": int(
                (cache_report or {}).get(
                    "forced_compute_steps", transformer_compute_steps
                )
            ),
            "reused_transformer_steps": int(
                (cache_report or {}).get(
                    "reuse_candidate_steps",
                    num_inference_steps - transformer_compute_steps,
                )
            ),
            "prediction_reuse_mode": (
                "linear-first-order" if enable_denoiser_step_reuse else None
            ),
            "selected_attention_is_approximate": selected_attention_is_approximate,
            "resolution_fixed": optimization_profile
            in {
                "one-shot-exact",
                "one-shot-fast",
                "one-shot-aggressive",
                "one-shot-80",
                "one-shot-80-quality",
            },
        },
        "guidance_scale": guidance_scale,
        "output_type": output_type,
        "warmup_ms": warmup_times,
        "wall_ms": wall_times,
        "cuda_event_ms": cuda_times,
        "wall_summary_ms": {
            "mean": statistics.fmean(wall_times),
            "median": statistics.median(wall_times),
            "p95": percentile(wall_times, 0.95),
            "min": min(wall_times),
        },
        "cuda_summary_ms": {
            "mean": statistics.fmean(cuda_times),
            "median": statistics.median(cuda_times),
            "p95": percentile(cuda_times, 0.95),
            "min": min(cuda_times),
        },
        "prompt_cuda_ms": prompt_cuda_times,
        "prompt_cuda_summary_ms": {
            "mean": statistics.fmean(prompt_cuda_times),
            "median": statistics.median(prompt_cuda_times),
            "p95": percentile(prompt_cuda_times, 0.95),
            "min": min(prompt_cuda_times),
        },
        "vae_encode_cuda_ms": vae_encode_cuda_times,
        "vae_encode_cuda_summary_ms": (
            {
                "mean": statistics.fmean(vae_encode_cuda_times),
                "median": statistics.median(vae_encode_cuda_times),
                "p95": percentile(vae_encode_cuda_times, 0.95),
                "min": min(vae_encode_cuda_times),
            }
            if vae_encode_cuda_times
            else None
        ),
        "overlapped_preparation_cuda_ms": overlapped_preparation_cuda_times,
        "overlapped_preparation_cuda_summary_ms": (
            {
                "mean": statistics.fmean(overlapped_preparation_cuda_times),
                "median": statistics.median(overlapped_preparation_cuda_times),
                "p95": percentile(overlapped_preparation_cuda_times, 0.95),
                "min": min(overlapped_preparation_cuda_times),
            }
            if overlapped_preparation_cuda_times
            else None
        ),
        "cache": cache_report,
        "cache_validation": cache_validation,
        "attention_backend": attention_backend,
        "attention_selection_report": attention_selection_report,
        "klein_cuda_ops": {
            "requested": enable_klein_cuda_ops,
            "loaded": klein_cuda_ops_loaded,
            "transformer_patched": klein_cuda_ops_patched,
        },
        "sm120_nvfp4_extension": {
            "requested": requested_nvfp4_gemm_backend == "cutlass-sm120",
            "loaded": sm120_nvfp4_extension is not None,
            "setup_error": sm120_nvfp4_setup_error,
            "tiles": (
                [
                    "128x128x256",
                    "128x128x128",
                    "128x64x256",
                    "128x64x128",
                    "128x32x256",
                    "128x32x128",
                ]
                if sm120_nvfp4_extension is not None
                else None
            ),
            "epilogue": "dynamic-scale" if sm120_nvfp4_extension is not None else None,
        },
        "direct_nvfp4_dispatch": {
            "requested": enable_direct_nvfp4_dispatch,
            "report": direct_nvfp4_report,
            "trace_validation": direct_dispatch_validation,
        },
        "static_transformer_activation_scales": {
            "requested": enable_static_transformer_activation_scales,
            "calibration": static_activation_calibration_report,
            "latent_parity": static_activation_parity,
        },
        "prequant_qkv_fusion": prequant_qkv_fusion_report,
        "fused_qkv_packing": {
            "requested": enable_fused_qkv_packing,
            "report": fused_qkv_packing_report,
        },
        "input_preprocessed_outside_timing": True,
        "prompt_and_vae_encode_overlapped": overlap_preparation,
        "prompt_embeds_precomputed_outside_timing": False,
        "noise_precomputed_outside_timing": False,
        "unique_prompt_count": unique_prompt_count,
        "expected_full_compute_nvfp4_gemms": expected_full_compute_gemms if not enable_cache else None,
        "expected_executed_nvfp4_gemms": (
            expected_executed_nvfp4_gemms if not enable_cache else None
        ),
        "expected_attention_invocations": (
            expected_attention_invocations if not enable_cache else None
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if save_outputs_dir:
        if output_type == "pil":
            validation_images = measured_images
            print("validation_render=reusing_measured_pil_outputs timing_excluded_file_io=true")
        else:
            print("validation_render=start timing_excluded=true prompt_cache=fresh")
            pipe._prompt_cache.clear()
            validation_images = []
            with torch.inference_mode():
                for i, prompt in enumerate(prompt_list, start=1):
                    image = call_pipeline(prompt, output_type_override="pil")[0]
                    validation_images.append(image)
                    print(f"validation_render=[{i:02d}/{len(prompt_list):02d}] prompt={prompt!r}")
        run_dir = _save_benchmark_images_after_timing(
            images=validation_images,
            prompts=prompt_list,
            base_dir=Path(save_outputs_dir),
            volume=MODEL_VOLUME,
            volume_mount=VOLUME_MOUNT,
            benchmark_result=result,
        )
        print(f"Saved {len(validation_images)} validation images and benchmark.json to {run_dir} (post-timed)")
    # Modal's local client does not need torch installed to deserialize results.
    # Round-trip through JSON to remove TorchVersion and any other subclasses.
    return json.loads(json.dumps(result))


@APP.local_entrypoint()
def main(
    use_taef2: bool = True,
    measured_runs: int = 20,
    enable_cache: bool = False,
    cache_steps_mask: str = "1111",
    cache_steps_computation_policy: str = "dynamic",
    backend: str = "torchao-nvfp4",
    enable_klein_cuda_ops: bool = False,
    enable_direct_nvfp4_dispatch: bool = True,
    enable_static_transformer_activation_scales: bool = False,
    enable_full_denoise_compile: bool = False,
    enable_denoiser_step_reuse: bool = False,
    fuse_qkv_before_quantization: bool = False,
    enable_fused_qkv_packing: bool = False,
    nvfp4_gemm_backend: str = "torch-scaled-mm",
    attention_backend_request: str = "auto",
    text_encoder_backend: str = "bf16",
    nvfp4_text_encoder_dir: str = "/mnt/klein4B-assets/Qwen3-4B-NVFP4",
    compile_text_encoder: bool = True,
    enable_torch_compile: bool = True,
    overlap_preparation: bool = False,
    optimization_profile: str = "baseline",
    nsight_capture: bool = False,
    enable_internal_profiler_validation: bool = True,
) -> None:
    benchmark.remote(
        use_taef2=use_taef2,
        measured_runs=measured_runs,
        enable_cache=enable_cache,
        cache_steps_mask=cache_steps_mask,
        cache_steps_computation_policy=cache_steps_computation_policy,
        backend=backend,
        enable_klein_cuda_ops=enable_klein_cuda_ops,
        enable_direct_nvfp4_dispatch=enable_direct_nvfp4_dispatch,
        enable_static_transformer_activation_scales=enable_static_transformer_activation_scales,
        enable_full_denoise_compile=enable_full_denoise_compile,
        enable_denoiser_step_reuse=enable_denoiser_step_reuse,
        fuse_qkv_before_quantization=fuse_qkv_before_quantization,
        enable_fused_qkv_packing=enable_fused_qkv_packing,
        nvfp4_gemm_backend=nvfp4_gemm_backend,
        attention_backend_request=attention_backend_request,
        text_encoder_backend=text_encoder_backend,
        nvfp4_text_encoder_dir=nvfp4_text_encoder_dir,
        compile_text_encoder=compile_text_encoder,
        enable_torch_compile=enable_torch_compile,
        overlap_preparation=overlap_preparation,
        optimization_profile=optimization_profile,
        nsight_capture=nsight_capture,
        enable_internal_profiler_validation=enable_internal_profiler_validation,
    )
