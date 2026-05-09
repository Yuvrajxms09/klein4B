"""FluxRT-aligned helpers (shared-memory tensors, aspect-safe crop)."""

from .crop_maximal_rectangle import crop_maximal_rectangle
from .shared_tensor import SharedTensor

__all__ = ["SharedTensor", "crop_maximal_rectangle"]
