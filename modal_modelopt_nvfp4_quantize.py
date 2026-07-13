"""Quantize FLUX.2 Klein 4B to NVFP4 with Model Optimizer and export HF weights.

This script uses the Model Optimizer diffusion PTQ flow directly:

- load the Klein diffusers pipeline from the existing HF checkpoint on the Modal volume
- quantize the transformer with `modelopt.torch.quantization.mtq.quantize`
- export a Hugging Face checkpoint with `export_hf_checkpoint`

The calibration loop runs Klein img2img on a small fixed prompt/image set so the
quantizers are calibrated on the same shape the runtime cares about.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
import types
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import modal


APP = modal.App("klein4b-modelopt-nvfp4-quantize")
VOLUME_MOUNT = "/mnt/klein4B-assets"
MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)

WORKSPACE = Path(__file__).resolve().parent.parent
KLEIN_ROOT = Path(__file__).resolve().parent
MODELOPT_ROOT = WORKSPACE / "Model-Optimizer"
DIFFUSERS_ROOT = WORKSPACE / "diffusers"

DEFAULT_PROMPTS: tuple[str, ...] = (
    "from top angle",
    "on a bike",
    "on a mountain road",
    "in a rainy city at night",
    "under golden sunset light",
    "in a snowy forest",
    "with a cinematic teal and orange look",
    "as a realistic product photo",
)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("build-essential", "git", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel ninja",
        "pip install --index-url https://download.pytorch.org/whl/cu128 "
        "torch==2.8.0 torchvision==0.23.0",
    )
    .pip_install(
        "accelerate",
        "einops",
        "huggingface_hub",
        "numpy",
        "nvidia-ml-py>=12",
        "omegaconf>=2.3.0",
        "packaging",
        "pillow",
        "pulp<4.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "regex",
        "requests",
        "rich",
        "safetensors",
        "scipy",
        "tqdm",
        "transformers>=4.56,<5.13",
    )
    .add_local_dir(str(MODELOPT_ROOT), remote_path="/root/Model-Optimizer", copy=True)
    .add_local_dir(str(DIFFUSERS_ROOT), remote_path="/root/diffusers", copy=True)
    .add_local_dir(str(KLEIN_ROOT), remote_path="/root/klein4B", copy=True)
    .run_commands("pip install -e '/root/diffusers' --no-deps")
    .run_commands("pip install -e '/root/Model-Optimizer[hf]' --no-deps")
)


def _read_safetensors_metadata(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _maybe_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class QuantizeConfig:
    model_dir: str = f"{VOLUME_MOUNT}/FLUX.2-klein-4B"
    image_path: str = f"{VOLUME_MOUNT}/calib/blue_car.jpeg"
    output_dir: str = f"{VOLUME_MOUNT}/quantized_klein4b/nvfp4_hf"
    dtype: str = "bfloat16"
    height: int = 576
    width: int = 384
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    calib_batch_size: int = 1
    calib_prompts: int = 8
    calib_repeat: int = 1
    use_taef2: bool = True
    taef2_cache_dir: str = f"{VOLUME_MOUNT}/taef2"
    save_modelopt_state: bool = True
    enable_swizzle_layout: bool = True
    enable_layerwise_quant_metadata: bool = True
    padding_strategy: str = "row_col"
    max_shard_size: str = "10GB"
    local_files_only: bool = True


def _quantize_cfg():
    import modelopt.torch.quantization as mtq

    return copy.deepcopy(mtq.NVFP4_FP8_MHA_CONFIG)


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60 * 2,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
)
def quantize_klein_nvfp4(
    config: QuantizeConfig = QuantizeConfig(),
) -> dict[str, Any]:
    import torch
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from PIL import Image

    sys.path.insert(0, "/root/Model-Optimizer")
    sys.path.insert(0, "/root/diffusers/src")
    sys.path.insert(0, "/root/klein4B")

    from klein_pipeline import Flux2KleinPipeline
    from taef2_vae import replace_pipeline_vae_with_taef2

    process_started = time.perf_counter()
    current_stage = {"name": "startup"}
    heartbeat_stop = threading.Event()

    def set_stage(name: str) -> None:
        current_stage["name"] = name

    def heartbeat() -> None:
        while not heartbeat_stop.wait(30.0):
            print(
                f"[modelopt heartbeat +{time.perf_counter() - process_started:9.2f}s]"
                f" stage={current_stage['name']}",
                flush=True,
            )

    threading.Thread(target=heartbeat, name="modelopt-heartbeat", daemon=True).start()

    def log(message: str) -> None:
        elapsed = time.perf_counter() - process_started
        memory = ""
        if torch.cuda.is_available():
            memory = (
                f" cuda_alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB"
                f" cuda_reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB"
                f" cuda_max={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"
            )
        print(f"[modelopt +{elapsed:9.2f}s stage={current_stage['name']}]" f"{memory} {message}", flush=True)

    def sync_log(message: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        log(message)

    def tensor_info(name: str, value: Any) -> str:
        if not isinstance(value, torch.Tensor):
            return f"{name}=<{type(value).__name__}>"
        return (
            f"{name}.shape={tuple(value.shape)} dtype={value.dtype}"
            f" device={value.device} contiguous={value.is_contiguous()}"
        )

    log("function entered")
    log(f"config={json.dumps(asdict(config), sort_keys=True)}")
    log(f"torch={torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")

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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_dir = Path(config.model_dir)
    image_path = Path(config.image_path)
    output_dir = Path(config.output_dir)
    set_stage("validate_inputs")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    log(f"validated model_dir={model_dir} image={image_path} output={output_dir}")

    set_stage("enable_modelopt_checkpointing")
    log("enabling ModelOpt Hugging Face checkpoint integration")
    mto.enable_huggingface_checkpointing()
    log("ModelOpt Hugging Face checkpoint integration enabled")

    dtype = getattr(torch, config.dtype)
    set_stage("load_pipeline")
    log(f"loading pipeline dtype={dtype} local_files_only={config.local_files_only}")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        local_files_only=config.local_files_only,
    )
    sync_log("pipeline loaded")
    pipe.set_progress_bar_config(disable=True)
    set_stage("move_pipeline_to_cuda")
    log("moving pipeline to CUDA")
    pipe = pipe.to(device="cuda")
    sync_log(f"pipeline moved to CUDA execution_device={pipe._execution_device}")
    if config.use_taef2:
        set_stage("load_taef2")
        log(f"loading TAEF2 cache_dir={config.taef2_cache_dir}")
        replace_pipeline_vae_with_taef2(
            pipe,
            cache_dir=config.taef2_cache_dir,
            device="cuda",
            dtype=dtype,
        )
        sync_log(f"TAEF2 installed vae={type(pipe.vae).__name__}")
    else:
        log(f"TAEF2 disabled; vae={type(pipe.vae).__name__}")

    set_stage("prepare_calibration")
    log("loading ModelOpt quantization configuration")
    quant_cfg = _quantize_cfg()
    log(
        f"quant config algorithm={quant_cfg.get('algorithm')}"
        f" rules={len(quant_cfg.get('quant_cfg', []))}"
    )
    input_image = Image.open(image_path).convert("RGB").resize((config.width, config.height))
    log(f"calibration image prepared size={input_image.size} mode={input_image.mode}")
    prompts = list(DEFAULT_PROMPTS[: max(1, config.calib_prompts)])
    if len(prompts) < config.calib_prompts:
        prompts = (prompts * ((config.calib_prompts + len(prompts) - 1) // len(prompts)))[: config.calib_prompts]
    log(f"calibration prompts={len(prompts)} steps={config.num_inference_steps}")

    # Materialize the exact transformer inputs once. ModelOpt's calibration loop
    # should call the module being quantized directly; running the full pipeline
    # from inside it would recalibrate Qwen, VAE, and scheduler code repeatedly.
    calibration_inputs: list[dict[str, Any]] = []
    original_run_denoiser = pipe._run_denoiser

    def capture_run_denoiser(self, **kwargs):
        call_index = len(calibration_inputs) + 1
        log(
            f"[capture {call_index:03d}] denoiser start"
            f" {tensor_info('hidden_states', kwargs.get('hidden_states'))}"
        )
        calibration_inputs.append({key: value for key, value in kwargs.items()})
        started = time.perf_counter()
        try:
            output = original_run_denoiser(**kwargs)
        except Exception:
            log(f"[capture {call_index:03d}] denoiser FAILED")
            traceback.print_exc()
            raise
        sync_log(
            f"[capture {call_index:03d}] denoiser done"
            f" elapsed_ms={(time.perf_counter() - started) * 1000.0:.1f}"
        )
        return output

    pipe._run_denoiser = types.MethodType(capture_run_denoiser, pipe)
    set_stage("capture_transformer_inputs")
    log("preparing transformer calibration inputs; pipeline forwards begin")
    for index, prompt in enumerate(prompts, start=1):
        before_calls = len(calibration_inputs)
        log(f"[inputs {index:02d}/{len(prompts):02d}] start prompt={prompt!r}")
        started = time.perf_counter()
        try:
            pipe(
                prompt=prompt,
                image=input_image,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                output_type="latent",
            )
        except Exception:
            log(f"[inputs {index:02d}] FAILED during pipeline capture")
            traceback.print_exc()
            raise
        sync_log(
            f"[inputs {index:02d}/{len(prompts):02d}] done"
            f" prompt={prompt!r} transformer_calls={len(calibration_inputs) - before_calls}"
            f" elapsed_ms={(time.perf_counter() - started) * 1000.0:.1f}"
        )
    pipe._run_denoiser = original_run_denoiser

    if not calibration_inputs:
        raise RuntimeError("Calibration input capture produced no transformer calls")
    log(f"captured total transformer inputs={len(calibration_inputs)}")
    first = calibration_inputs[0]
    log(
        "first calibration input: "
        + ", ".join(
            tensor_info(name, first.get(name))
            for name in ("hidden_states", "timestep", "encoder_hidden_states", "txt_ids", "img_ids")
        )
        + f" context={first.get('context')!r}"
    )

    call_count = 0
    quant_cfg_summary = {
        "algorithm": quant_cfg.get("algorithm"),
        "num_quant_rules": len(quant_cfg.get("quant_cfg", [])),
    }

    def forward_loop(_mod: torch.nn.Module) -> None:
        nonlocal call_count
        set_stage("modelopt_forward_loop")
        total = len(calibration_inputs) * config.calib_repeat
        log(
            f"ModelOpt forward_loop entered module={type(_mod).__name__}"
            f" parameters={sum(1 for _ in _mod.parameters())} forwards={total}"
        )
        started = time.perf_counter()
        for repeat in range(config.calib_repeat):
            for index, inputs in enumerate(calibration_inputs, start=1):
                started_forward = time.perf_counter()
                log(f"[calib {call_count + 1:03d}/{total:03d}] start repeat={repeat + 1} input={index}")
                try:
                    with _mod.cache_context(inputs["context"]):
                        _mod(
                            hidden_states=inputs["hidden_states"],
                            timestep=inputs["timestep"] / 1000,
                            guidance=None,
                            encoder_hidden_states=inputs["encoder_hidden_states"],
                            txt_ids=inputs["txt_ids"],
                            img_ids=inputs["img_ids"],
                            joint_attention_kwargs=inputs["joint_attention_kwargs"],
                            return_dict=False,
                        )
                    sync_log(
                        f"[calib {call_count + 1:03d}/{total:03d}] done"
                        f" elapsed_ms={(time.perf_counter() - started_forward) * 1000.0:.1f}"
                    )
                except Exception:
                    log(f"[calib {call_count + 1:03d}] FAILED")
                    traceback.print_exc()
                    raise
                call_count += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        sync_log(f"ModelOpt forward_loop completed elapsed_ms={elapsed_ms:.1f}")

    set_stage("mtq_quantize")
    log("calling mtq.quantize: module replacement and calibration begin")
    try:
        mtq.quantize(pipe.transformer, quant_cfg, forward_loop)
    except Exception:
        log("mtq.quantize FAILED")
        traceback.print_exc()
        raise
    sync_log("mtq.quantize completed")

    if config.save_modelopt_state:
        set_stage("save_modelopt_state")
        log("saving ModelOpt state")
        state_path = output_dir / "transformer_modelopt_state.pt"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        mto.save(pipe.transformer, str(state_path))
        log(f"ModelOpt state saved path={state_path} bytes={state_path.stat().st_size}")

    set_stage("export_hf_checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    log("exporting Hugging Face checkpoint")
    export_hf_checkpoint(
        pipe,
        dtype=dtype,
        export_dir=output_dir,
        components=["transformer"],
        enable_swizzle_layout=config.enable_swizzle_layout,
        enable_layerwise_quant_metadata=config.enable_layerwise_quant_metadata,
        padding_strategy=config.padding_strategy,
        max_shard_size=config.max_shard_size,
    )
    log("Hugging Face checkpoint export completed")

    safetensors_files = sorted(output_dir.rglob("*.safetensors"))
    if not safetensors_files:
        raise RuntimeError(f"No safetensors files were exported to {output_dir}")
    log(f"export contains safetensors_files={len(safetensors_files)}")

    metadata_by_file = {
        str(path.relative_to(output_dir)): _read_safetensors_metadata(path) for path in safetensors_files
    }
    metadata_text = json.dumps(metadata_by_file, sort_keys=True).upper()
    if "NVFP4" not in metadata_text:
        raise RuntimeError("Exported checkpoint does not advertise NVFP4 in safetensors metadata")
    log("export metadata confirms NVFP4")

    manifest = {
        "backend": "nvidia-modelopt",
        "quantization": "NVFP4",
        "base_model": config.model_dir,
        "image_path": config.image_path,
        "output_dir": config.output_dir,
        "dtype": config.dtype,
        "modelopt_config": quant_cfg_summary,
        "safetensors": [str(path.relative_to(output_dir)) for path in safetensors_files],
        "metadata": metadata_by_file,
        "config": asdict(config),
    }
    set_stage("commit_volume")
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log("manifest written; committing Modal volume")
    MODEL_VOLUME.commit()
    sync_log("Modal volume commit completed")
    heartbeat_stop.set()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


@APP.local_entrypoint()
def main() -> None:
    quantize_klein_nvfp4.remote()
