from importlib import import_module

from .shared_tensor import SharedTensor

__all__ = ["SharedTensor", "crop_maximal_rectangle"]


def __getattr__(name):
    if name != "crop_maximal_rectangle":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.crop_maximal_rectangle")
    return module.crop_maximal_rectangle
