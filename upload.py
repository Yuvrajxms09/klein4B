"""Populate the Modal volume used by the Klein benchmark."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import modal


APP = modal.App("klein4b-model-uploader")
MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=True)
VOLUME_MOUNT = "/models"

image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub")
    .add_local_file(
        str(Path(__file__).resolve().parent / "blue_car_resize.jpeg"),
        remote_path="/root/blue_car_resize.jpeg",
    )
)


@APP.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
)
def upload_model() -> None:
    from huggingface_hub import snapshot_download

    target = f"{VOLUME_MOUNT}/FLUX.2-klein-4B"
    snapshot_download(
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        local_dir=target,
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.git*", "*.gitattributes", ".git*"],
    )
    MODEL_VOLUME.commit()
    print(f"Model uploaded to {target}")


@APP.function(
    image=image,
    timeout=600,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
)
def upload_input_image() -> None:
    target_dir = Path(VOLUME_MOUNT) / "calib"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "blue_car.jpeg"
    shutil.copy2("/root/blue_car_resize.jpeg", target)
    MODEL_VOLUME.commit()
    print(f"Input image uploaded to {target}")


@APP.local_entrypoint()
def main() -> None:
    upload_model.remote()
    upload_input_image.remote()
