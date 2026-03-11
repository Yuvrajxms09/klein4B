"""
TAEF2 lightweight VAE support for Klein/FLUX.2 pipelines.

Based on the wrapper pattern from https://huggingface.co/madebyollin/taef2 (Tiny
AutoEncoder for FLUX.2). We add latent_dist.mode() for pipeline sample_mode="argmax"
and copy the original VAE's BN running stats so normalize/denormalize stay correct.

Use replace_pipeline_vae_with_taef2(pipe, ...) after loading the pipeline to swap
the default VAE with a TAEF2 wrapper (same latent API, faster encode/decode).
"""

from __future__ import annotations

import importlib.util
import sys
import urllib.request
from pathlib import Path
from typing import Any

import safetensors.torch as stt
import torch
from diffusers.utils.accelerate_utils import apply_forward_hook


def _download_if_missing(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    request = urllib.request.Request(url, headers={"User-Agent": "klein4b-taef2/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    output_path.write_bytes(data)
    return output_path


def ensure_taef2_artifacts(
    cache_dir: Path | str,
    taesd_py_path: str | Path | None = None,
    taef2_weight_path: str | Path | None = None,
) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if taesd_py_path is not None and str(taesd_py_path).strip():
        taesd_path = Path(taesd_py_path).expanduser().resolve()
        if not taesd_path.exists():
            raise FileNotFoundError(f"TAEF2 taesd.py not found: {taesd_path}")
    else:
        taesd_path = cache_dir / "taesd.py"
        _download_if_missing(
            "https://raw.githubusercontent.com/madebyollin/taesd/refs/heads/main/taesd.py",
            taesd_path,
        )

    if taef2_weight_path is not None and str(taef2_weight_path).strip():
        weight_path = Path(taef2_weight_path).expanduser().resolve()
        if not weight_path.exists():
            raise FileNotFoundError(f"TAEF2 weight not found: {weight_path}")
    else:
        weight_path = cache_dir / "taef2.safetensors"
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="madebyollin/taef2",
                filename="taef2.safetensors",
                local_dir=str(cache_dir),
            )
        except Exception:
            _download_if_missing(
                "https://huggingface.co/madebyollin/taef2/resolve/main/taef2.safetensors",
                weight_path,
            )

    if not weight_path.exists():
        raise FileNotFoundError(f"TAEF2 weight not found: {weight_path}")
    return taesd_path, weight_path


def _load_taesd_class(taesd_py_path: Path):
    spec = importlib.util.spec_from_file_location("taesd_dynamic", str(taesd_py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {taesd_py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "TAESD"):
        raise AttributeError(f"Module {taesd_py_path} does not export TAESD")
    return module.TAESD


def _convert_diffusers_sd_to_taesd(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        parts = key.split(".")
        if len(parts) < 3:
            out[key] = value
            continue
        encdec, _layers, index, *suffix = parts
        if not index.isdigit():
            out[key] = value
            continue
        offset = 1 if encdec == "decoder" else 0
        mapped_key = ".".join([encdec, str(int(index) + offset), *suffix])
        out[mapped_key] = value
    return out


class _DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def build_taef2_diffusers_vae(
    taesd_py_path: Path,
    taef2_weight_path: Path,
    device: str | torch.device,
    dtype: torch.dtype,
    bn_channels: int = 128,
    batch_norm_eps: float = 0.0,
) -> torch.nn.Module:
    taesd_cls = _load_taesd_class(taesd_py_path)

    class DiffusersTAEF2Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dtype = dtype
            self.taesd = taesd_cls(
                encoder_path=None,
                decoder_path=None,
                latent_channels=32,
                arch_variant="flux_2",
            ).to(dtype=self.dtype)
            self.taesd.load_state_dict(
                _convert_diffusers_sd_to_taesd(stt.load_file(str(taef2_weight_path), device="cpu"))
            )
            self.bn = torch.nn.BatchNorm2d(int(bn_channels), affine=False, eps=float(batch_norm_eps))
            self.config = _DotDict(batch_norm_eps=float(self.bn.eps))

        @apply_forward_hook
        def encode(self, x: torch.Tensor):
            encoded = self.taesd.encoder(x.to(dtype=self.dtype).mul(0.5).add_(0.5)).to(dtype=x.dtype)
            latent_dist = _DotDict(sample=lambda generator=None: encoded, mode=lambda: encoded)
            return _DotDict(latent_dist=latent_dist)

        @apply_forward_hook
        def decode(self, x: torch.Tensor, return_dict: bool = True):
            decoded = self.taesd.decoder(x.to(dtype=self.dtype)).mul(2).sub_(1).clamp_(-1, 1).to(dtype=x.dtype)
            if return_dict:
                return {"sample": decoded}
            return (decoded,)

    return DiffusersTAEF2Wrapper().to(device=device).eval().requires_grad_(False)


def replace_pipeline_vae_with_taef2(
    pipe: Any,
    *,
    cache_dir: str | Path = ".cache/taef2",
    taesd_py_path: str | Path | None = None,
    taef2_weight_path: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """
    Replace the pipeline's VAE with a TAEF2 lightweight wrapper (same API, faster).

    Call this after loading the pipeline (e.g. Flux2KleinPipeline.from_pretrained(...)).
    Pipeline's vae_scale_factor is unchanged (set at init), so no pipeline code changes.

    BN running stats are copied from the original VAE so latent normalize/denormalize
    stays correct.

    Returns the same pipeline for chaining.
    """
    if device is None:
        device = next(pipe.transformer.parameters()).device
    if dtype is None:
        dtype = getattr(pipe.vae, "dtype", next(pipe.transformer.parameters()).dtype)

    bn_channels = 128
    batch_norm_eps = 0.0
    if hasattr(pipe.vae, "bn") and hasattr(pipe.vae.bn, "running_mean"):
        bn_channels = int(pipe.vae.bn.running_mean.shape[0])
        batch_norm_eps = float(getattr(pipe.vae.bn, "eps", 0.0))
    elif hasattr(pipe.vae, "config") and getattr(pipe.vae.config, "batch_norm_eps", None) is not None:
        batch_norm_eps = float(pipe.vae.config.batch_norm_eps)

    taesd_path, weight_path = ensure_taef2_artifacts(
        cache_dir=cache_dir,
        taesd_py_path=taesd_py_path,
        taef2_weight_path=taef2_weight_path,
    )

    wrapper = build_taef2_diffusers_vae(
        taesd_py_path=taesd_path,
        taef2_weight_path=weight_path,
        device=device,
        dtype=dtype,
        bn_channels=bn_channels,
        batch_norm_eps=batch_norm_eps,
    )

    if hasattr(pipe.vae.bn, "running_mean") and hasattr(pipe.vae.bn, "running_var"):
        with torch.no_grad():
            wrapper.bn.running_mean.copy_(pipe.vae.bn.running_mean)
            wrapper.bn.running_var.copy_(pipe.vae.bn.running_var)

    pipe.vae = wrapper
    return pipe
