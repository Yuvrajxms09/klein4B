"""FluxRT stream processor package."""

from importlib import import_module

__all__ = ["StreamProcessor", "OutputSchedulerSubprocess"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.stream_processor")
    return getattr(module, name)
