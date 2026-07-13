"""Restore the existing Klein ModelOpt checkpoint and export it for SGLang."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal


APP = modal.App("klein4b-modelopt-nvfp4-export")
VOLUME_MOUNT = "/mnt/klein4B-assets"
MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)

WORKSPACE = Path(__file__).resolve().parent.parent
KLEIN_ROOT = Path(__file__).resolve().parent
MODELOPT_ROOT = WORKSPACE / "Model-Optimizer"
DIFFUSERS_ROOT = WORKSPACE / "diffusers"

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
        "transformers",
    )
    .add_local_dir(str(MODELOPT_ROOT), remote_path="/root/Model-Optimizer", copy=True)
    .add_local_dir(str(DIFFUSERS_ROOT), remote_path="/root/diffusers", copy=True)
    .add_local_dir(str(KLEIN_ROOT), remote_path="/root/klein4B", copy=True)
    .run_commands("pip install -e '/root/Model-Optimizer[hf]' --no-deps")
)


def _read_safetensors_metadata(path: Path) -> dict:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
)
def export_existing_checkpoint(
    *,
    model_dir: str = f"{VOLUME_MOUNT}/FLUX.2-klein-4B",
    checkpoint_path: str = f"{VOLUME_MOUNT}/quantized_klein4b/nvfp4/transformer_modelopt.pt",
    output_dir: str = f"{VOLUME_MOUNT}/quantized_klein4b/nvfp4_sglang",
) -> dict:
    import torch

    sys.path.insert(0, "/root/Model-Optimizer")
    sys.path.insert(0, "/root/diffusers/src")
    sys.path.insert(0, "/root/klein4B")

    import modelopt.torch.opt as mto
    from modelopt.torch.export import export_hf_checkpoint
    from klein_pipeline import Flux2KleinPipeline

    model_path = Path(model_dir)
    checkpoint = Path(checkpoint_path)
    destination = Path(output_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    restored = mto.restore(pipe.transformer, checkpoint)
    if restored is not pipe.transformer:
        pipe.transformer = restored

    destination.mkdir(parents=True, exist_ok=True)
    export_hf_checkpoint(
        pipe,
        dtype=torch.bfloat16,
        export_dir=destination,
        components=["transformer"],
        enable_swizzle_layout=True,
        enable_layerwise_quant_metadata=True,
        padding_strategy="row_col",
        max_shard_size="10GB",
    )

    safetensors_files = sorted(destination.rglob("*.safetensors"))
    if not safetensors_files:
        raise RuntimeError(f"ModelOpt export produced no safetensors under {destination}")

    metadata_by_file = {str(path.relative_to(destination)): _read_safetensors_metadata(path) for path in safetensors_files}
    metadata_text = json.dumps(metadata_by_file, sort_keys=True).upper()
    config_files = sorted(destination.rglob("*.json"))
    config_payloads = {}
    for path in config_files:
        try:
            config_payloads[str(path.relative_to(destination))] = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    config_text = json.dumps(config_payloads, sort_keys=True).upper()
    if "NVFP4" not in metadata_text and "NVFP4" not in config_text:
        raise RuntimeError("Exported checkpoint does not advertise NVFP4 in safetensors metadata or config")

    manifest = {
        "backend": "nvidia-modelopt",
        "quantization": "NVFP4",
        "source_checkpoint": checkpoint_path,
        "base_model": model_dir,
        "output_dir": output_dir,
        "safetensors": [str(path.relative_to(destination)) for path in safetensors_files],
        "metadata": metadata_by_file,
        "configs": [str(path.relative_to(destination)) for path in config_files],
    }
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    MODEL_VOLUME.commit()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


@APP.local_entrypoint()
def main() -> None:
    export_existing_checkpoint.remote()
