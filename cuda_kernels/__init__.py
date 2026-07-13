import importlib
import logging


logger = logging.getLogger(__name__)


def is_loaded() -> bool:
    try:
        _ = torch_ops_namespace()
        return True
    except Exception:
        return False


def torch_ops_namespace():
    import torch

    ns = getattr(torch.ops, "klein_cuda", None)
    if ns is None:
        raise RuntimeError("torch.ops.klein_cuda not found; build/load extension first")
    return ns


def load_compiled_extension(module_name: str = "klein_cuda_ext") -> None:
    logger.info("loading Klein CUDA extension module=%s", module_name)
    importlib.import_module(module_name)
    logger.info("Klein CUDA extension loaded module=%s", module_name)
