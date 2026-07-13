"""Validate and benchmark the locally exported ModelOpt NVFP4 Klein transformer in SGLang.

This is intentionally separate from the TorchAO benchmark. It validates the exact
ModelOpt HF export first, then exercises SGLang's Flux2 Klein img2img path with
that export supplied through ``transformer_weights_path``.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import modal


APP = modal.App("klein4b-sglang-modelopt-validation")
VOLUME_MOUNT = "/mnt/klein4B-assets"
MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)
REPO_ROOT = Path(__file__).resolve().parent.parent
SGLANG_ROOT = REPO_ROOT / "sglang"
TRANSFORMERS_ROOT = REPO_ROOT / "transformers"
KLEIN_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "libnuma-dev", "libopenblas-dev")
    .run_commands("python -m pip install --upgrade pip wheel setuptools ninja")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu130 "
        "torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0"
    )
    # Install runtime dependencies without installing a released SGLang wheel.
    # Released SGLang versions currently require huggingface-hub<1, while this
    # local Transformers checkout requires huggingface-hub>=1.5.
    .uv_pip_install(
        "accelerate",
        "aiohttp",
        "apache-tvm-ffi==0.1.11",
        "addict==2.4.0",
        "anthropic>=0.20.0",
        "av==16.1.0",
        "blobfile==3.0.0",
        "build",
        "cache-dit==1.3.0",
        "cloudpickle==3.1.2",
        "compressed-tensors",
        "cuda-python>=13.0",
        "datasets",
        "distro",
        "easydict",
        "einops",
        "fastapi",
        "flashinfer_python[cu13]==0.6.14",
        "huggingface-hub>=1.5.0,<2.0",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "IPython",
        "interegular",
        "kernels>=0.14.1,<0.15",
        "llguidance>=0.7.11,<0.8.0",
        "mistral_common>=1.11.5",
        "msgspec",
        "modelscope",
        "moviepy>=2.0.0",
        "numpy",
        "nvidia-cutlass-dsl[cu13]==4.5.2",
        "nvidia-ml-py",
        "nvidia-mathdx==25.6.0",
        "orjson",
        "openai==2.6.1",
        "openai-harmony==0.0.4",
        "outlines==0.1.11",
        "packaging",
        "partial_json_parser",
        "pillow",
        "prometheus-client>=0.20.0",
        "pybase64",
        "pydantic",
        "py-spy",
        "pyzmq>=25.1.2",
        "PyYAML==6.0.1",
        "regex",
        "remote-pdb==2.1.0",
        "requests",
        "runai_model_streamer>=0.15.7",
        "safetensors",
        "scikit-image==0.25.2",
        "scipy",
        "sentencepiece",
        "sgl-deep-gemm==0.1.4",
        "sglang-kernel==0.4.4",
        "setproctitle",
        "smg-grpc-servicer>=0.5.0",
        "soundfile==0.13.1",
        "tiktoken",
        "tokenizers",
        "tilelang==0.1.11",
        "timm==1.0.16",
        "torch-memory-saver>=0.0.9.post1",
        "torchao==0.17.0",
        "tqdm",
        "uvicorn",
        "uvloop",
        "watchfiles",
        "trimesh>=4.0.0",
        "vsa==0.0.4",
        "xatlas",
        "xgrammar==0.2.1",
        "zstandard",
        "opencv-python-headless==4.10.0.84",
    )
    .add_local_dir(str(SGLANG_ROOT), remote_path="/root/sglang", copy=True)
    .add_local_dir(str(TRANSFORMERS_ROOT), remote_path="/root/transformers", copy=True)
    .add_local_dir(str(KLEIN_ROOT), remote_path="/root/klein4B", copy=True)
    .env({"PYTHONPATH": "/root/sglang/python:/root/transformers/src:/root/klein4B"})
)


def _validate_export(export_dir: Path) -> dict:
    config_path = export_dir / "transformer" / "config.json"
    if not config_path.is_file():
        config_path = export_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"ModelOpt config.json not found under {export_dir}")

    config = json.loads(config_path.read_text())
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise RuntimeError("Export has no quantization_config")
    if quant.get("quant_method") != "modelopt":
        raise RuntimeError(f"Unexpected quant_method: {quant.get('quant_method')!r}")
    if str(quant.get("quant_algo", "")).upper() != "NVFP4":
        raise RuntimeError(f"Export is not NVFP4: {quant.get('quant_algo')!r}")

    shards = sorted((export_dir / "transformer").glob("*.safetensors"))
    if not shards:
        shards = sorted(export_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors weights found under {export_dir}")
    return {
        "config": str(config_path),
        "quant_method": quant["quant_method"],
        "quant_algo": quant["quant_algo"],
        "producer": quant.get("producer"),
        "safetensors": [str(path) for path in shards],
    }


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
)
def validate(
    *,
    model_path: str = f"{VOLUME_MOUNT}/FLUX.2-klein-4B",
    transformer_weights_path: str = f"{VOLUME_MOUNT}/quantized_klein4b/nvfp4_hf",
    image_path: str = f"{VOLUME_MOUNT}/calib/blue_car.jpeg",
    height: int = 576,
    width: int = 384,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    warmup_runs: int = 2,
    measured_runs: int = 3,
) -> dict:
    repo_root = Path("/root/sglang/python")
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, "/root/transformers/src")
    os.environ["SGLANG_DIFFUSION_FP4_GEMM_BACKEND"] = "flashinfer_trtllm"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    export_info = _validate_export(Path(transformer_weights_path))
    print("MODEL_OPT_EXPORT", json.dumps(export_info, indent=2, sort_keys=True), flush=True)
    if not Path(model_path).is_dir():
        raise FileNotFoundError(model_path)
    if not Path(image_path).is_file():
        raise FileNotFoundError(image_path)

    import torch
    import transformers
    from sglang.multimodal_gen import DiffGenerator
    from sglang.multimodal_gen.runtime.server_args import Backend

    torch.set_grad_enabled(False)
    print(f"TRANSFORMERS_RUNTIME version={transformers.__version__} path={transformers.__file__}", flush=True)
    generator_kwargs = {
        "model_path": model_path,
        "transformer_weights_path": transformer_weights_path,
        "backend": Backend.from_string("sglang"),
        "num_gpus": 1,
        "enable_torch_compile": False,
        "dit_cpu_offload": False,
        "attention_backend": "fa",
        "output_path": "/tmp/klein4b-sglang-modelopt",
    }
    print("SGLANG_LOAD", json.dumps(generator_kwargs, default=str, sort_keys=True), flush=True)
    prompts = ["from top angle", "on a bike", "on a mountain road", "in a rainy city at night"]
    timings_ms: list[float] = []

    def sample(prompt: str, seed: int):
        return {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "num_frames": 1,
            "return_frames": True,
            "save_output": False,
            "progressive_mode": "fullres",
            "progressive_levels": 1,
            "progressive_delta": 0.10,
            "image_path": image_path,
        }

    with DiffGenerator.from_pretrained(local_mode=True, **generator_kwargs) as generator:
        print("SGLANG_LOAD_OK generator initialized", flush=True)
        for index in range(warmup_runs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            generator.generate(sampling_params_kwargs=sample(prompts[index % len(prompts)], index))
            torch.cuda.synchronize()
            print(f"WARMUP {index + 1}/{warmup_runs} {(time.perf_counter() - started) * 1000:.1f} ms", flush=True)

        for index in range(measured_runs):
            prompt = prompts[index % len(prompts)]
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = generator.generate(sampling_params_kwargs=sample(prompt, 100 + index))
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            if result is None:
                raise RuntimeError("SGLang returned no result")
            timings_ms.append(elapsed)
            print(f"MEASURED {index + 1}/{measured_runs} {elapsed:.1f} ms prompt={prompt!r}", flush=True)

    summary = {
        "backend": "sglang",
        "quantization": "ModelOpt-NVFP4",
        "model_path": model_path,
        "transformer_weights_path": transformer_weights_path,
        "image_path": image_path,
        "resolution": [height, width],
        "num_inference_steps": num_inference_steps,
        "wall_ms": timings_ms,
        "wall_summary_ms": {
            "mean": statistics.fmean(timings_ms),
            "median": statistics.median(timings_ms),
            "min": min(timings_ms),
        },
        "export": export_info,
        "includes_sglang_img2img_pipeline": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@APP.local_entrypoint()
def main() -> None:
    validate.remote()
