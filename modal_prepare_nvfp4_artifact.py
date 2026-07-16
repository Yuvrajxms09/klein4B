"""Create and publish the pre-quantized TorchAO checkpoint used by the benchmark.

The first run quantizes the Klein transformer and the active 27-layer Qwen text
encoder, writes an atomic artifact to the ``klein4B-assets`` Modal volume,
validates that both components reload as pre-quantized models, and uploads the
same artifact to a Hugging Face model repository. Later benchmark runs
load these packed weights directly; activation quantization remains dynamic.

Run once before ``fastest_script_prequantized_nvfp4.py``:

    modal run modal_prepare_nvfp4_artifact.py

Use ``--force`` only when intentionally rebuilding the artifact after changing
the model or software versions.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import modal


APP = modal.App("klein4b-prepare-torchao-nvfp4")
HF_SECRET = modal.Secret.from_name("huggingface-secret")
MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)
VOLUME_MOUNT = "/mnt/klein4B-assets"

EXPECTED_TRANSFORMER_LINEARS = 109
QWEN_LAYERS = 27
EXPECTED_QWEN_LINEARS = QWEN_LAYERS * 7
DEFAULT_HF_REPO_ID = "Yuvrajxms09/klein-torchao-artifacts"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libopenblas-dev")
    .env({"TORCH_CUDA_ARCH_LIST": "12.0", "USE_HUB_KERNELS": "NO"})
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu130 "
        "torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0"
    )
    .pip_install("numpy")
    .pip_install("pillow")
    .pip_install("fastapi")
    .pip_install("uvicorn")
    .pip_install("accelerate")
    .pip_install("safetensors")
    .pip_install("huggingface_hub")
    .pip_install("importlib-metadata", "filelock", "httpx", "regex", "requests")
    .pip_install("einops")
    .pip_install("pybase64")
    .pip_install("cache-dit")
    .pip_install("kernels==0.16.0")
    .pip_install("ninja", "setuptools", "wheel")
    .run_commands(
        "pip install -U git+https://github.com/huggingface/transformers.git@"
        "63f32a8782cb70da3365acab16f2b67947737985"
    )
    .run_commands("pip install torchao==0.17.0")
    .run_commands("pip install mslk-cuda==1.1.0")
    .add_local_dir(
        str(_workspace_root()),
        remote_path="/root/klein4B",
        copy=True,
    )
    .add_local_dir(
        str(_workspace_root().parent / "diffusers"),
        remote_path="/root/diffusers",
        copy=True,
    )
)


def _configure_imports() -> None:
    repo_root = Path("/root/klein4B")
    diffusers_root = Path("/root/diffusers/src")
    os.chdir(repo_root)
    for path in (repo_root, diffusers_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _nvfp4_config():
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

    return NVFP4DynamicActivationNVFP4WeightConfig(
        use_triton_kernel=True,
        use_dynamic_per_tensor_scale=True,
    )


def _count_nvfp4_linears(module) -> int:
    import torch
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    return sum(
        1
        for child in module.modules()
        if isinstance(child, torch.nn.Linear)
        and isinstance(child.weight, NVFP4Tensor)
    )


def _require_nvfp4_linears(module, *, expected: int, component: str) -> None:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    observed = _count_nvfp4_linears(module)
    if observed != expected:
        raise RuntimeError(
            f"Unexpected {component} NVFP4 coverage: expected={expected}, "
            f"observed={observed}"
        )

    invalid_activation_configs = []
    for name, child in module.named_modules():
        weight = getattr(child, "weight", None)
        if weight is None or not isinstance(weight, NVFP4Tensor):
            continue
        activation_config = getattr(weight, "act_quant_kwargs", None)
        if activation_config is None or not getattr(
            activation_config,
            "use_dynamic_per_tensor_scale",
            False,
        ):
            invalid_activation_configs.append(name)
    if invalid_activation_configs:
        raise RuntimeError(
            f"{component} contains NVFP4 weights without dynamic activation "
            f"quantization: {invalid_activation_configs[:5]}"
        )


def _require_prequantized_loader(model, *, component: str) -> None:
    quantizer = getattr(model, "hf_quantizer", None)
    if quantizer is None or not getattr(quantizer, "pre_quantized", False):
        raise RuntimeError(f"{component} did not use the pre-quantized load path")


def _clear_cuda_cache() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _build_transformer(*, source_dir: Path, output_dir: Path) -> None:
    import torch
    from diffusers import Flux2Transformer2DModel, TorchAoConfig

    print("transformer_export=quantizing")
    transformer = Flux2Transformer2DModel.from_pretrained(
        source_dir,
        subfolder="transformer",
        quantization_config=TorchAoConfig(_nvfp4_config()),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    _require_nvfp4_linears(
        transformer,
        expected=EXPECTED_TRANSFORMER_LINEARS,
        component="transformer",
    )
    print("transformer_export=saving format=pytorch-bin")
    transformer.save_pretrained(
        output_dir,
        safe_serialization=False,
        max_shard_size="4GB",
    )
    del transformer
    _clear_cuda_cache()


def _build_text_encoder(*, source_dir: Path, output_dir: Path) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    text_encoder_dir = source_dir / "text_encoder"
    config = AutoConfig.from_pretrained(text_encoder_dir, local_files_only=True)
    original_layers = int(config.num_hidden_layers)
    if original_layers < QWEN_LAYERS:
        raise RuntimeError(
            f"Qwen config has {original_layers} layers; Klein requires {QWEN_LAYERS}"
        )

    for field_name in ("layer_types", "mlp_layer_types"):
        layer_types = getattr(config, field_name, None)
        if layer_types is None:
            continue
        if len(layer_types) < QWEN_LAYERS:
            raise RuntimeError(
                f"Qwen config field {field_name} has only {len(layer_types)} entries"
            )
        setattr(config, field_name, list(layer_types[:QWEN_LAYERS]))
    config.num_hidden_layers = QWEN_LAYERS
    config.validate()

    print(
        f"text_encoder_export=quantizing source_layers={original_layers} "
        f"saved_layers={QWEN_LAYERS}"
    )
    text_encoder = AutoModelForCausalLM.from_pretrained(
        text_encoder_dir,
        config=config,
        quantization_config=TransformersTorchAoConfig(
            _nvfp4_config(),
            modules_to_not_convert=["lm_head"],
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        local_files_only=True,
    )
    if len(text_encoder.model.layers) != QWEN_LAYERS:
        raise RuntimeError("Qwen model topology does not match the reduced config")
    _require_nvfp4_linears(
        text_encoder.model,
        expected=EXPECTED_QWEN_LINEARS,
        component="Qwen decoder",
    )
    print("text_encoder_export=saving format=safetensors")
    text_encoder.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    del text_encoder
    _clear_cuda_cache()


def _package_versions() -> dict[str, str]:
    import diffusers
    import transformers

    packages = ("torch", "torchao", "huggingface_hub")
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    versions["transformers"] = transformers.__version__
    versions["transformers_revision"] = "63f32a8782cb70da3365acab16f2b67947737985"
    versions["diffusers"] = diffusers.__version__
    return versions


def _write_metadata(*, artifact_dir: Path, source_dir: Path) -> None:
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": str(source_dir),
        "quantization": {
            "format": "torchao-nvfp4",
            "weights": "nvfp4",
            "activations": "dynamic-nvfp4",
            "use_dynamic_per_tensor_scale": True,
            "use_triton_kernel": True,
        },
        "transformer": {
            "serialization": "pytorch-bin",
            "nvfp4_linear_count": EXPECTED_TRANSFORMER_LINEARS,
        },
        "text_encoder": {
            "serialization": "safetensors",
            "decoder_layers": QWEN_LAYERS,
            "nvfp4_linear_count": EXPECTED_QWEN_LINEARS,
            "lm_head_quantized": False,
        },
        "versions": _package_versions(),
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "README.md").write_text(
        "---\n"
        "base_model: black-forest-labs/FLUX.2-klein-4B\n"
        "library_name: diffusers\n"
        "pipeline_tag: image-to-image\n"
        "tags:\n"
        "  - torchao\n"
        "  - nvfp4\n"
        "  - quantized\n"
        "  - blackwell\n"
        "---\n\n"
        "# FLUX.2 Klein 4B TorchAO NVFP4\n\n"
        "Pre-quantized TorchAO NVFP4 weights for the FLUX.2 Klein 4B transformer "
        "and the active 27-layer Qwen encoder. Activations remain dynamically "
        "quantized at inference time.\n\n"
        "The original [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) "
        "repository is required for the tokenizer, scheduler, and remaining "
        "pipeline components. Loading and benchmark code is available in the "
        "[`optimized-nvfp4-115ms`](https://github.com/Yuvrajxms09/klein4B/tree/optimized-nvfp4-115ms) "
        "branch. See `manifest.json` for serialization details and pinned package "
        "versions.\n",
        encoding="utf-8",
    )


def _validate_artifact(artifact_dir: Path) -> dict[str, object]:
    import torch
    import torchao.prototype.mx_formats  # noqa: F401
    from diffusers import Flux2Transformer2DModel
    from transformers import AutoModelForCausalLM

    print("artifact_validation=loading_prequantized_components")
    transformer = Flux2Transformer2DModel.from_pretrained(
        artifact_dir,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=False,
    )
    text_encoder = AutoModelForCausalLM.from_pretrained(
        artifact_dir / "text_encoder",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    transformer.to("cuda")
    text_encoder.to("cuda")

    _require_prequantized_loader(transformer, component="transformer")
    _require_prequantized_loader(text_encoder, component="Qwen text encoder")
    _require_nvfp4_linears(
        transformer,
        expected=EXPECTED_TRANSFORMER_LINEARS,
        component="transformer",
    )
    _require_nvfp4_linears(
        text_encoder.model,
        expected=EXPECTED_QWEN_LINEARS,
        component="Qwen decoder",
    )
    if len(text_encoder.model.layers) != QWEN_LAYERS:
        raise RuntimeError("Reloaded Qwen model does not contain exactly 27 layers")

    result = {
        "prequantized_load": True,
        "transformer_nvfp4_linears": EXPECTED_TRANSFORMER_LINEARS,
        "text_encoder_nvfp4_linears": EXPECTED_QWEN_LINEARS,
        "text_encoder_layers": QWEN_LAYERS,
    }
    del transformer, text_encoder
    _clear_cuda_cache()
    print(f"artifact_validation={json.dumps(result, sort_keys=True)}")
    return result


def _publish_atomically(*, staging_dir: Path, artifact_dir: Path, force: bool) -> None:
    backup_dir = artifact_dir.with_name(f".{artifact_dir.name}.backup-{uuid.uuid4().hex}")
    if artifact_dir.exists():
        if not force:
            raise FileExistsError(f"Artifact already exists: {artifact_dir}")
        artifact_dir.rename(backup_dir)

    try:
        staging_dir.rename(artifact_dir)
    except BaseException:
        if backup_dir.exists() and not artifact_dir.exists():
            backup_dir.rename(artifact_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def _resolve_hub_repo_id(requested_repo_id: str | None) -> str:
    if requested_repo_id:
        return requested_repo_id
    return DEFAULT_HF_REPO_ID


def _upload_artifact(
    *,
    artifact_dir: Path,
    repo_id: str,
    token: str,
    private: bool,
) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    visibility = "private" if api.model_info(repo_id=repo_id, token=token).private else "public"
    print(f"hub_upload=start repo_id={repo_id} visibility={visibility}")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=artifact_dir,
        commit_message="Add validated TorchAO NVFP4 deployment artifact",
    )
    print(f"hub_upload=complete commit={commit.oid}")
    return commit.oid


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=3 * 60 * 60,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
    secrets=[HF_SECRET],
)
def prepare(
    *,
    source_model_dir: str = "/mnt/klein4B-assets/FLUX.2-klein-4B",
    artifact_dir: str = "/mnt/klein4B-assets/FLUX.2-klein-4B-torchao-nvfp4",
    hf_repo_id: str | None = None,
    force: bool = False,
    upload_to_hub: bool = True,
    private: bool = False,
) -> dict[str, object]:
    _configure_imports()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    source_path = Path(source_model_dir).resolve()
    artifact_path = Path(artifact_dir).resolve()
    volume_path = Path(VOLUME_MOUNT).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source model not found: {source_path}")
    if not artifact_path.is_relative_to(volume_path):
        raise ValueError(f"artifact_dir must be inside the mounted volume: {VOLUME_MOUNT}")

    built = False
    if artifact_path.exists() and not force:
        print(f"artifact=existing path={artifact_path}")
        validation = _validate_artifact(artifact_path)
    else:
        staging_path = artifact_path.with_name(
            f".{artifact_path.name}.staging-{uuid.uuid4().hex}"
        )
        staging_path.mkdir(parents=True)
        try:
            _build_transformer(
                source_dir=source_path,
                output_dir=staging_path / "transformer",
            )
            _build_text_encoder(
                source_dir=source_path,
                output_dir=staging_path / "text_encoder",
            )
            _write_metadata(artifact_dir=staging_path, source_dir=source_path)
            validation = _validate_artifact(staging_path)
            _publish_atomically(
                staging_dir=staging_path,
                artifact_dir=artifact_path,
                force=force,
            )
            built = True
            MODEL_VOLUME.commit()
            print(f"artifact=committed path={artifact_path}")
        except BaseException:
            if staging_path.exists():
                shutil.rmtree(staging_path)
            raise

    upload_result = None
    if upload_to_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is missing from the huggingface-secret Modal secret")
        resolved_repo_id = _resolve_hub_repo_id(hf_repo_id)
        upload_result = {
            "repo_id": resolved_repo_id,
            "commit": _upload_artifact(
                artifact_dir=artifact_path,
                repo_id=resolved_repo_id,
                token=token,
                private=private,
            ),
        }

    result = {
        "artifact_dir": str(artifact_path),
        "built": built,
        "validation": validation,
        "hub": upload_result,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@APP.local_entrypoint()
def main(
    hf_repo_id: str | None = None,
    force: bool = False,
    upload_to_hub: bool = True,
    private: bool = False,
) -> None:
    prepare.remote(
        hf_repo_id=hf_repo_id,
        force=force,
        upload_to_hub=upload_to_hub,
        private=private,
    )
