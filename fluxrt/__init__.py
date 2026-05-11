"""FluxRT package."""

from importlib import import_module

__all__ = [
    "StreamProcessor",
    "SharedTensor",
    "crop_maximal_rectangle",
    "stream_processor",
    "utils",
]


def __getattr__(name):
    if name == "stream_processor":
        return import_module(f"{__name__}.stream_processor")
    if name == "utils":
        return import_module(f"{__name__}.utils")
    if name == "StreamProcessor":
        return import_module(f"{__name__}.stream_processor.stream_processor").StreamProcessor
    if name == "SharedTensor":
        return import_module(f"{__name__}.utils.shared_tensor").SharedTensor
    if name == "crop_maximal_rectangle":
        return import_module(f"{__name__}.utils.crop_maximal_rectangle").crop_maximal_rectangle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
