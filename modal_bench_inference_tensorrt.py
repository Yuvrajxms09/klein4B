"""
Modal export/debug job for a transformer-only TensorRT path for FLUX.2 Klein 4B.

This script intentionally keeps the surface area small:
- load the pipeline
- prepare one stable transformer export input set
- export the transformer
- try ONNX export
- try TensorRT / Torch-TensorRT compilation when available
- persist artifacts to the Modal volume

The goal is to make it easy to inspect what succeeded or failed before adding
runtime benchmarking or broader pipeline changes.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import modal


APP = modal.App("klein4b-inference-bench-tensorrt")
LOGGER = logging.getLogger("klein4b.tensorrt")


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
    .pip_install("accelerate")
    .pip_install("safetensors")
    .pip_install("huggingface_hub")
    .pip_install("transformers")
    .pip_install("einops")
    .pip_install("pybase64")
    .pip_install("onnx")
    .pip_install("onnxscript")
    .pip_install("tensorrt-rtx")
    .pip_install("cache-dit")
    .pip_install("ninja", "setuptools", "wheel")
    .run_commands("pip install -U torchao")
    .add_local_dir(_repo_root(), remote_path="/root/klein4B")
    .add_local_dir(str(Path(_repo_root()).parent / "diffusers"), remote_path="/root/diffusers")
    .add_local_dir(str(Path(_repo_root()).parent / "flux2"), remote_path="/root/flux2")
)


MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)
VOLUME_MOUNT = "/mnt/klein4B-assets"


@dataclass(frozen=True)
class ExportConfig:
    model_dir: str = "/mnt/klein4B-assets/FLUX.2-klein-4B"
    quantized_model_dir: str | None = None
    export_dir: str = "/mnt/klein4B-assets/tensorrt_artifacts"
    image_path: str = "/mnt/klein4B-assets/calib/blue_car.jpeg"
    height: int = 576
    width: int = 384
    guidance_scale: float = 1.0
    seed: int = 0
    precision: str = "bf16"
    shape_mode: str = "static"
    use_taef2: bool = True
    taef2_cache_dir: str = "/root/klein4B/.cache/taef2"
    local_files_only: bool = True


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _log(message: str) -> None:
    LOGGER.info(message)
    print(message, flush=True)


def _save_metadata(base_dir: Path, *, lines: list[str]) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    meta_path = base_dir / "export_debug.txt"
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta_path


def _load_transformer(model_dir: str, *, dtype, local_files_only: bool):
    from diffusers import Flux2Transformer2DModel

    _log(f"loading_transformer model_dir={model_dir} dtype={dtype} local_files_only={local_files_only}")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    transformer.eval()
    _log(f"transformer_loaded class={transformer.__class__.__name__}")
    return transformer


def _load_flux2_model(model_dir: str, *, device: str = "cuda"):
    from flux2.util import load_flow_model

    weight_path = Path(model_dir) / "transformer" / "flux-2-klein-4b.safetensors"
    if not weight_path.exists():
        weight_path = Path(model_dir) / "flux-2-klein-4b.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"flux2 weight file not found under {model_dir}")

    os.environ["KLEIN_4B_MODEL_PATH"] = str(weight_path)
    _log(f"loading_flux2_model weight_path={weight_path}")
    model = load_flow_model("flux.2-klein-4b", device=device)
    model.eval()
    _log(f"flux2_model_loaded class={model.__class__.__name__}")
    return model


def _load_pipeline(config: ExportConfig, *, dtype):
    import torch

    from klein_pipeline import Flux2KleinPipeline
    from taef2_vae import replace_pipeline_vae_with_taef2

    transformer = _load_transformer(
        config.model_dir,
        dtype=dtype,
        local_files_only=config.local_files_only,
    )

    _log("loading_pipeline")
    pipe = Flux2KleinPipeline.from_pretrained(
        config.model_dir,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=config.local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    _log("pipeline_moved_to_cuda")

    if config.use_taef2:
        _log(f"installing_taef2 cache_dir={config.taef2_cache_dir}")
        replace_pipeline_vae_with_taef2(pipe, cache_dir=config.taef2_cache_dir)
        if hasattr(pipe.vae, "taesd") and hasattr(pipe.vae.taesd, "decoder"):
            pipe.vae.taesd.decoder.to(memory_format=torch.channels_last)
            _log("taef2_decoder_channels_last=1")
    else:
        _log("using_native_vae")
        if hasattr(pipe.vae, "fuse_qkv_projections"):
            pipe.vae.fuse_qkv_projections()

    pipe.vae.to(memory_format=torch.channels_last)
    _log("vae_channels_last=1")
    return pipe


def _build_export_inputs(pipe: Any, config: ExportConfig):
    import torch
    from PIL import Image

    device = pipe._execution_device
    dummy_image = Image.new("RGB", (config.width, config.height), color=(128, 128, 128))
    prompt = "tensor rt export prompt"

    _log(
        "building_export_inputs "
        f"height={config.height} width={config.width} guidance_scale={config.guidance_scale} "
        f"seed={config.seed} device={device}"
    )

    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
        text_encoder_out_layers=(9, 18, 27),
    )
    _log(f"prompt_embeds shape={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}")
    _log(f"text_ids shape={tuple(text_ids.shape)} dtype={text_ids.dtype}")

    generator = torch.Generator(device=device).manual_seed(config.seed)
    image_tensor = pipe._preprocess_image_fast(
        dummy_image,
        height=config.height,
        width=config.width,
        resize_mode="crop",
    )
    _log(f"image_tensor shape={tuple(image_tensor.shape)} dtype={image_tensor.dtype}")
    image_latents, image_latent_ids = pipe.prepare_image_latents(
        images=[image_tensor],
        batch_size=1,
        generator=generator,
        device=device,
        dtype=pipe.vae.dtype,
        non_blocking_h2d=True,
    )
    _log(f"image_latents shape={tuple(image_latents.shape)} dtype={image_latents.dtype}")
    _log(f"image_latent_ids shape={tuple(image_latent_ids.shape)} dtype={image_latent_ids.dtype}")

    latents, latent_ids = pipe.prepare_latents(
        batch_size=1,
        num_latents_channels=pipe.transformer.config.in_channels // 4,
        height=config.height,
        width=config.width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    _log(f"latents shape={tuple(latents.shape)} dtype={latents.dtype}")
    _log(f"latent_ids shape={tuple(latent_ids.shape)} dtype={latent_ids.dtype}")

    timestep = torch.ones(latents.shape[0], device=device, dtype=prompt_embeds.dtype)
    guidance = torch.tensor([float(config.guidance_scale)], device=device, dtype=torch.float32)
    _log(f"timestep shape={tuple(timestep.shape)} dtype={timestep.dtype}")
    _log(f"guidance shape={tuple(guidance.shape)} dtype={guidance.dtype}")

    return {
        "hidden_states": latents,
        "timestep": timestep,
        "encoder_hidden_states": prompt_embeds,
        "txt_ids": text_ids,
        "img_ids": latent_ids,
        "guidance": guidance,
    }


def _resolve_tensorrt_module():
    candidates = ("tensorrt_rtx",)
    for name in candidates:
        try:
            module = __import__(name)
            _log(f"tensorrt_module_loaded={name}")
            return module
        except Exception as exc:
            _log(f"tensorrt_module_missing={name} error={exc!r}")
    return None


def _export_transformer(pipe: Any, config: ExportConfig) -> dict[str, str]:
    import torch

    export_dir = Path(config.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    _log(f"export_dir={export_dir}")
    precision = config.precision.lower().strip()
    if precision not in {"bf16", "fp16", "fp8", "fp4"}:
        raise ValueError(f"unsupported precision={config.precision!r}")
    model_dir_for_export = config.model_dir
    if precision == "fp4":
        if not config.quantized_model_dir:
            raise ValueError(
                "precision='fp4' requires quantized_model_dir to point at a pre-quantized "
                "NVFP4/FP4 checkpoint directory. The base bf16 Klein checkpoint is not enough."
            )
        model_dir_for_export = config.quantized_model_dir
        _log(f"using_quantized_model_dir={model_dir_for_export}")

    inputs = _build_export_inputs(pipe, config)
    export_transformer = _load_flux2_model(model_dir_for_export, device="cuda")
    _log(f"export_transformer_device={next(export_transformer.parameters()).device}")
    _log("export_target=_python_denoise_step")

    class PythonDenoiseExportWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(
            self,
            x: torch.Tensor,
            x_ids: torch.Tensor,
            timesteps: torch.Tensor,
            ctx: torch.Tensor,
            ctx_ids: torch.Tensor,
            guidance: torch.Tensor,
        ) -> torch.Tensor:
            return self.model._python_denoise_step(x, x_ids, timesteps, ctx, ctx_ids, guidance)

    wrapper = PythonDenoiseExportWrapper(export_transformer).eval()
    _log(f"export_wrapper_ready class={wrapper.__class__.__name__}")

    debug_lines = [
        f"model_dir={config.model_dir}",
        f"quantized_model_dir={config.quantized_model_dir}",
        f"model_dir_for_export={model_dir_for_export}",
        f"export_dir={export_dir}",
        f"height={config.height}",
        f"width={config.width}",
        f"guidance_scale={config.guidance_scale}",
        f"seed={config.seed}",
        f"precision={precision}",
        f"shape_mode={config.shape_mode}",
        f"use_taef2={config.use_taef2}",
        f"inputs={{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in inputs.items())}}}",
    ]

    artifacts: dict[str, str] = {}

    try:
        _log("starting_torch_export")
        ep = torch.export.export(
            wrapper,
            args=(),
            kwargs={
                "x": inputs["hidden_states"],
                "x_ids": inputs["img_ids"],
                "timesteps": inputs["timestep"],
                "ctx": inputs["encoder_hidden_states"],
                "ctx_ids": inputs["txt_ids"],
                "guidance": inputs["guidance"],
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        pt2_path = export_dir / "transformer_export.pt2"
        torch.export.save(ep, pt2_path)
        artifacts["pt2"] = str(pt2_path)
        _log(f"saved_export={pt2_path}")
    except Exception as exc:
        _log(f"torch_export_failed={exc!r}")

    try:
        _log("starting_onnx_export")
        onnx_path = export_dir / "transformer.onnx"
        onnx_input_names = ["x", "x_ids", "timesteps", "ctx", "ctx_ids", "guidance"]
        onnx_dynamic_axes = None if config.shape_mode == "static" else {k: {0: "batch"} for k in onnx_input_names}
        torch.onnx.export(
            wrapper,
            (
                inputs["hidden_states"],
                inputs["img_ids"],
                inputs["timestep"],
                inputs["encoder_hidden_states"],
                inputs["txt_ids"],
                inputs["guidance"],
            ),
            onnx_path.as_posix(),
            input_names=onnx_input_names,
            output_names=["sample"],
            opset_version=18,
            dynamic_axes=onnx_dynamic_axes,
        )
        artifacts["onnx"] = str(onnx_path)
        _log(f"saved_onnx={onnx_path}")
    except Exception as exc:
        _log("onnx_export_failed=1")
        _log(f"onnx_export_error={exc!r}")

    trt = _resolve_tensorrt_module()
    if trt is not None:
        try:
            _log("starting_tensorrt_rtx_engine_build")
            onnx_path = Path(artifacts.get("onnx", export_dir / "transformer.onnx"))
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX file not found for engine build: {onnx_path}")
            engine_path = export_dir / "transformer.trt"
            import subprocess

            cmd = [
                "tensorrt_rtx",
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
            ]
            if precision in {"bf16", "fp16", "fp8", "fp4"}:
                cmd.insert(1, f"--{precision}")
            _log(f"running={' '.join(map(str, cmd))}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            _log(f"tensorrt_rtx_returncode={proc.returncode}")
            if proc.stdout:
                _log(f"tensorrt_rtx_stdout={proc.stdout.strip()}")
            if proc.stderr:
                _log(f"tensorrt_rtx_stderr={proc.stderr.strip()}")
            if proc.returncode != 0:
                raise RuntimeError(f"tensorrt_rtx failed with exit code {proc.returncode}")
            artifacts["engine"] = str(engine_path)
            _log(f"saved_engine={engine_path}")
        except Exception as exc:
            _log(f"tensorrt_engine_build_failed={exc!r}")
    else:
        _log("tensorrt_rtx_not_available=1")

    meta_path = _save_metadata(export_dir, lines=debug_lines + [f"artifacts={artifacts}"])
    artifacts["debug_log"] = str(meta_path)
    _log(f"saved_debug_log={meta_path}")
    return artifacts


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
)
def export_transformer(
    *,
    model_dir: str = ExportConfig.model_dir,
    export_dir: str = ExportConfig.export_dir,
    image_path: str = ExportConfig.image_path,
    height: int = ExportConfig.height,
    width: int = ExportConfig.width,
    guidance_scale: float = ExportConfig.guidance_scale,
    seed: int = ExportConfig.seed,
    precision: str = ExportConfig.precision,
    shape_mode: str = ExportConfig.shape_mode,
    use_taef2: bool = ExportConfig.use_taef2,
    taef2_cache_dir: str = ExportConfig.taef2_cache_dir,
    local_files_only: bool = ExportConfig.local_files_only,
) -> dict[str, str]:
    import torch

    _configure_logging()
    _log("starting_tensorRT_export_job")
    _log(
        "config "
        f"model_dir={model_dir} export_dir={export_dir} image_path={image_path} "
        f"height={height} width={width} guidance_scale={guidance_scale} seed={seed} "
        f"precision={precision} shape_mode={shape_mode} use_taef2={use_taef2} "
        f"local_files_only={local_files_only}"
    )

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not Path(image_path).exists():
        _log(f"warning_image_path_missing={image_path}")

    try:
        torch.set_grad_enabled(False)
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
    flux2_src = Path("/root/flux2/src")
    if str(flux2_src) not in sys.path:
        sys.path.insert(0, str(flux2_src))
    diffusers_root = Path("/root/diffusers/src")
    if str(diffusers_root) not in sys.path:
        sys.path.insert(0, str(diffusers_root))

    config = ExportConfig(
        model_dir=model_dir,
        export_dir=export_dir,
        image_path=image_path,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        seed=seed,
        precision=precision,
        shape_mode=shape_mode,
        use_taef2=use_taef2,
        taef2_cache_dir=taef2_cache_dir,
        local_files_only=local_files_only,
    )

    pipe = _load_pipeline(config, dtype=torch.bfloat16)
    artifacts = _export_transformer(pipe, config)
    _log(f"artifact_summary={artifacts}")
    return artifacts


@APP.local_entrypoint()
def main(
    precision: str = ExportConfig.precision,
    shape_mode: str = ExportConfig.shape_mode,
    use_taef2: bool = ExportConfig.use_taef2,
    model_dir: str = ExportConfig.model_dir,
    export_dir: str = ExportConfig.export_dir,
) -> None:
    result = export_transformer.remote(
        precision=precision,
        shape_mode=shape_mode,
        use_taef2=use_taef2,
        model_dir=model_dir,
        export_dir=export_dir,
    )
    _log(f"remote_result={result}")
