"""
RIFE frame interpolation — same batching logic as FluxRT ``ModelInferenceSubprocess.interpolate_frames``.

Weights: point ``weights_path`` at ``flownet.safetensors`` (e.g. ``RIFE-safetensors/flownet.safetensors``).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def _ifnet_cls():
    """Resolve IFNet whether ``klein4B`` is on sys.path as a folder or as an installed package."""
    try:
        from rife_ifnet import IFNet as _IFNet

        return _IFNet
    except ImportError:
        pass
    try:
        from klein4B.rife_ifnet import IFNet as _IFNet

        return _IFNet
    except ImportError:
        pass
    _path = Path(__file__).resolve().parent / "rife_ifnet.py"
    if not _path.is_file():
        raise ImportError(f"Cannot find rife_ifnet.py next to interpolation.py ({_path})")
    name = "_klein4b_rife_ifnet"
    spec = importlib.util.spec_from_file_location(name, _path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Invalid import spec for {_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.IFNet


IFNet = _ifnet_cls()


def load_rife_ifnet(
    weights_path: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float16,
) -> IFNet:
    model = IFNet()
    sd = load_file(weights_path)
    incomp = model.load_state_dict(sd, strict=False)
    if incomp.missing_keys:
        logger.warning("RIFE weights missing %d keys (first few): %s", len(incomp.missing_keys), incomp.missing_keys[:8])
    if incomp.unexpected_keys:
        logger.warning(
            "RIFE weights had %d unexpected keys (ignored; first few): %s",
            len(incomp.unexpected_keys),
            incomp.unexpected_keys[:8],
        )
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def expand_batch_with_rife(
    previous_frame: torch.Tensor,
    new_frame: torch.Tensor,
    model: IFNet,
    interpolation_exp: int,
) -> torch.Tensor:
    """
    ``previous_frame`` / ``new_frame``: (1, 3, H, W), same device/dtype as ``model``.
    Returns tensor of shape ``(2**interpolation_exp, 3, H, W)`` — interpolated steps then the new frame
    (FluxRT: ``frames[1:]`` after loop), matching one decode batch for ``interpolation_exp >= 1``.
    """
    if interpolation_exp < 1:
        return new_frame

    frames = torch.cat([previous_frame, new_frame], dim=0)
    with torch.no_grad():
        for _ in range(interpolation_exp):
            b = frames.size(0)
            prevs = frames[:-1]
            nexts = frames[1:]
            mids = model(torch.cat([prevs, nexts], dim=1))
            h, w = frames.shape[2:]
            new_frames = torch.empty(2 * b - 1, 3, h, w, device=frames.device, dtype=frames.dtype)
            new_frames[0::2] = frames
            new_frames[1::2] = mids
            frames = new_frames
    return frames[1:]
