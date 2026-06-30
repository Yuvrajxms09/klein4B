import os.path as osp
import sys
from importlib import import_module

_lp = osp.abspath(osp.join(osp.dirname(__file__), "..", "LivePortrait-code"))
LIVEPORTRAIT_AVAILABLE = osp.isdir(_lp)

if LIVEPORTRAIT_AVAILABLE and _lp not in sys.path:
    sys.path.insert(0, _lp)
    import src as _lp_src

    sys.modules.setdefault("liveportrait", _lp_src)

__all__ = [
    "LIVEPORTRAIT_AVAILABLE",
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
    if name == "LIVEPORTRAIT_AVAILABLE":
        return LIVEPORTRAIT_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
