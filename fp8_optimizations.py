from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from diffusers.utils import logging

FP8_CLAMP_MAX = 448.0
logger = logging.get_logger(__name__)


@torch.compile(dynamic=True)
def _fp8_quantize_compile(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Fused by torch.compile: scale + clamp + cast.
    return (x / scale.to(x.dtype)).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(torch.float8_e4m3fn)


def _fp8_quantize_triton(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Triton fused quantize kernel: x/scale -> clamp -> fp8 cast.
    Falls back to compile path when Triton is unavailable.
    """
    try:
        import triton
        import triton.language as tl
    except Exception as exc:
        logger.info("fp8 triton quantize unavailable, fallback to compile: %r", exc)
        return _fp8_quantize_compile(x, scale)

    if not x.is_cuda:
        logger.info("fp8 triton quantize received non-cuda tensor, fallback to compile")
        return _fp8_quantize_compile(x, scale)

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig, dtype=torch.float8_e4m3fn)
    n_elements = x_contig.numel()
    scale_f32 = float(scale.detach().float().item())
    inv_scale = 1.0 / max(scale_f32, 1e-12)

    @triton.jit
    def _quant_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        inv_scale: tl.constexpr,
        clamp_max: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x_val = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = x_val * inv_scale
        y = tl.minimum(tl.maximum(y, -clamp_max), clamp_max)
        tl.store(out_ptr + offs, y, mask=mask)

    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _quant_kernel[grid](
        x_contig,
        out,
        n_elements,
        inv_scale=inv_scale,
        clamp_max=FP8_CLAMP_MAX,
        BLOCK_SIZE=block_size,
    )
    return out


class FP8Linear(nn.Module):
    """
    FP8 linear layer using torch._scaled_mm.

    Expects:
    - weight stored in float8_e4m3fn
    - calibrated input_scale and weight_scale buffers
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        quantize_backend: str = "compile",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_backend = quantize_backend
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("input_scale", input_scale.detach().clone().reshape(()), persistent=True)
        self.register_buffer("weight_scale", weight_scale.detach().clone().reshape(()), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        try:
            if self.quantize_backend == "triton":
                x_fp8 = _fp8_quantize_triton(x_2d, self.input_scale)
            else:
                x_fp8 = _fp8_quantize_compile(x_2d, self.input_scale)
        except Exception as exc:
            logger.warning("fp8 quantize backend=%s failed, fallback to compile: %r", self.quantize_backend, exc)
            x_fp8 = _fp8_quantize_compile(x_2d, self.input_scale)
        out = torch._scaled_mm(
            x_fp8,
            self.weight.T,
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            out_dtype=x.dtype,
        )
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)


@dataclass
class FP8ApplyResult:
    total_linear: int
    replaced: int
    skipped_no_scale: int
    skipped_dtype: int
    missing_scale_modules: list[str]
    skipped_dtype_modules: list[str]


def _looks_like_fp8_weight(weight: torch.Tensor) -> bool:
    return weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def _resolve_scale(
    scales: dict[str, torch.Tensor] | None,
    module_name: str,
    kind: str,
    default: float | None,
) -> torch.Tensor | None:
    if scales is not None:
        key = f"{module_name}.{kind}"
        if key in scales:
            t = scales[key]
            if t.numel() != 1:
                logger.warning(
                    "scale tensor for %s has numel=%d (expected scalar); current path uses scalar-only scales",
                    key,
                    t.numel(),
                )
            return scales[key]
    if default is not None:
        return torch.tensor(default, dtype=torch.float32)
    return None


def _collect_linear_module_names(module: nn.Module, prefix: str = "") -> list[str]:
    names: list[str] = []
    for child_name, child in module.named_children():
        fq_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            names.append(fq_name)
        names.extend(_collect_linear_module_names(child, prefix=fq_name))
    return names


def _normalize_scale_key(state_key: str) -> str:
    # Handle checkpoint keys like "transformer.x_embedder.weight_scale".
    parts = state_key.split(".")
    if len(parts) >= 3 and parts[0] == "transformer":
        return ".".join(parts[1:])
    return state_key


def _find_scale_file_candidates(checkpoint_dir: str) -> list[str]:
    root = Path(checkpoint_dir)
    if not root.exists():
        return []
    candidates: list[str] = []
    # Prefer transformer subfolder if present in diffusers format.
    transformer_dir = root / "transformer"
    scan_roots = [transformer_dir, root] if transformer_dir.exists() else [root]
    patterns = ("*.safetensors", "*.bin", "*.pt")
    for scan_root in scan_roots:
        for pat in patterns:
            for p in scan_root.glob(pat):
                candidates.append(str(p))
    # Keep deterministic ordering.
    return sorted(set(candidates))


def _load_state_dict_file(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file  # type: ignore

        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    return {}


def load_scales_from_checkpoint(
    checkpoint_dir_or_file: str,
    *,
    strict: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Auto-load calibrated scales from a local HF-style checkpoint.

    Supports:
    - single file (.safetensors/.bin/.pt)
    - model directory (including diffusers layout with transformer/ subfolder)
    """
    path = Path(checkpoint_dir_or_file)
    files: list[str]
    if path.is_file():
        files = [str(path)]
    else:
        files = _find_scale_file_candidates(str(path))
    logger.info("fp8 scale loading: source=%s files=%d", checkpoint_dir_or_file, len(files))

    scales: dict[str, torch.Tensor] = {}
    loaded_files = 0
    for f in files:
        try:
            sd = _load_state_dict_file(f)
            loaded_files += 1
        except Exception as exc:
            logger.warning("failed to load scale candidate file %s: %r", f, exc)
            continue
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.endswith(".input_scale") or k.endswith(".weight_scale"):
                scales[_normalize_scale_key(k)] = v
    logger.info("fp8 scale loading: loaded_files=%d scales_found=%d", loaded_files, len(scales))

    if strict and not scales:
        raise RuntimeError(
            f"No calibrated FP8 scales found under: {checkpoint_dir_or_file}. "
            "Expected keys ending with .input_scale/.weight_scale."
        )
    return scales


def load_scales_from_hf_repo(
    repo_id: str,
    *,
    revision: str | None = None,
    local_dir: str | None = None,
    strict: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Download (or reuse local cache) and parse calibrated scales from an HF repo.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for load_scales_from_hf_repo") from exc

    target_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.safetensors",
            "*.bin",
            "*.pt",
            "transformer/*.safetensors",
            "transformer/*.bin",
            "transformer/*.pt",
        ],
    )
    logger.info("fp8 scale loading: downloaded hf repo=%s revision=%s dir=%s", repo_id, revision, target_dir)
    return load_scales_from_checkpoint(target_dir, strict=strict)


def _replace_linears_recursive(
    module: nn.Module,
    *,
    prefix: str,
    scales: dict[str, torch.Tensor] | None,
    default_input_scale: float | None,
    default_weight_scale: float | None,
    allow_cast_non_fp8_weights: bool,
    include_modules: set[str] | None,
    quantize_backend: str,
    missing_scale_modules: list[str],
    skipped_dtype_modules: list[str],
) -> FP8ApplyResult:
    total_linear = 0
    replaced = 0
    skipped_no_scale = 0
    skipped_dtype = 0

    for child_name, child in module.named_children():
        fq_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            total_linear += 1
            if include_modules is not None and fq_name not in include_modules:
                continue

            if not _looks_like_fp8_weight(child.weight):
                if not allow_cast_non_fp8_weights:
                    skipped_dtype += 1
                    skipped_dtype_modules.append(fq_name)
                    continue

            input_scale = _resolve_scale(scales, fq_name, "input_scale", default_input_scale)
            weight_scale = _resolve_scale(scales, fq_name, "weight_scale", default_weight_scale)
            if input_scale is None or weight_scale is None:
                skipped_no_scale += 1
                missing_scale_modules.append(fq_name)
                continue

            fp8_linear = FP8Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                input_scale=input_scale,
                weight_scale=weight_scale,
                quantize_backend=quantize_backend,
            )
            with torch.no_grad():
                fp8_linear.weight.copy_(child.weight.to(torch.float8_e4m3fn))
                if child.bias is not None:
                    fp8_linear.bias.copy_(child.bias)

            setattr(module, child_name, fp8_linear)
            replaced += 1
            continue

        sub = _replace_linears_recursive(
            child,
            prefix=fq_name,
            scales=scales,
            default_input_scale=default_input_scale,
            default_weight_scale=default_weight_scale,
            allow_cast_non_fp8_weights=allow_cast_non_fp8_weights,
            include_modules=include_modules,
            quantize_backend=quantize_backend,
            missing_scale_modules=missing_scale_modules,
            skipped_dtype_modules=skipped_dtype_modules,
        )
        total_linear += sub.total_linear
        replaced += sub.replaced
        skipped_no_scale += sub.skipped_no_scale
        skipped_dtype += sub.skipped_dtype

    return FP8ApplyResult(
        total_linear=total_linear,
        replaced=replaced,
        skipped_no_scale=skipped_no_scale,
        skipped_dtype=skipped_dtype,
        missing_scale_modules=missing_scale_modules,
        skipped_dtype_modules=skipped_dtype_modules,
    )


def apply_fp8_linears(
    transformer: nn.Module,
    *,
    scales: dict[str, torch.Tensor] | None = None,
    default_input_scale: float | None = None,
    default_weight_scale: float | None = None,
    allow_cast_non_fp8_weights: bool = False,
    include_modules: Iterable[str] | None = None,
    require_full_coverage: bool = False,
    quantize_backend: str = "compile",
) -> FP8ApplyResult:
    """
    Replace nn.Linear with FP8Linear when scales are available.

    Keep this opt-in and explicit because FP8 scale availability varies by checkpoint.
    """
    pre_linear_modules = set(_collect_linear_module_names(transformer))
    include_set = set(include_modules) if include_modules is not None else None
    if quantize_backend not in {"compile", "triton"}:
        raise ValueError("quantize_backend must be one of: {'compile', 'triton'}")
    missing_scale_modules: list[str] = []
    skipped_dtype_modules: list[str] = []
    result = _replace_linears_recursive(
        transformer,
        prefix="",
        scales=scales,
        default_input_scale=default_input_scale,
        default_weight_scale=default_weight_scale,
        allow_cast_non_fp8_weights=allow_cast_non_fp8_weights,
        include_modules=include_set,
        quantize_backend=quantize_backend,
        missing_scale_modules=missing_scale_modules,
        skipped_dtype_modules=skipped_dtype_modules,
    )
    logger.info(
        "fp8 apply result: total_linear=%d replaced=%d skipped_no_scale=%d skipped_dtype=%d backend=%s",
        result.total_linear,
        result.replaced,
        result.skipped_no_scale,
        result.skipped_dtype,
        quantize_backend,
    )
    if result.missing_scale_modules:
        logger.info("fp8 missing scales sample=%s", result.missing_scale_modules[:16])
    if result.skipped_dtype_modules:
        logger.info("fp8 skipped dtype sample=%s", result.skipped_dtype_modules[:16])
    if require_full_coverage:
        targets = include_set if include_set is not None else pre_linear_modules
        unresolved_targets = targets.intersection(set(missing_scale_modules + skipped_dtype_modules))
        if result.skipped_no_scale > 0 or result.skipped_dtype > 0 or unresolved_targets:
            details = []
            if missing_scale_modules:
                details.append(f"missing_scales={len(missing_scale_modules)}")
            if skipped_dtype_modules:
                details.append(f"dtype_skips={len(skipped_dtype_modules)}")
            if unresolved_targets:
                details.append(f"unresolved_targets={len(unresolved_targets)}")
            raise RuntimeError(
                "Full-model FP8 coverage requirement failed: " + ", ".join(details)
            )
    return result


def patch_flux2_rope_no_fp32_upcast() -> bool:
    """
    Monkey-patch Flux2 rotary function to avoid per-call fp32 upcast.
    Returns True when patch succeeds.
    """
    try:
        import diffusers.models.transformers.transformer_flux2 as tflux2
    except Exception as exc:
        logger.warning("rope patch import failed: %r", exc)
        return False
    original_apply_rotary_emb = tflux2.apply_rotary_emb

    def _apply_rotary_emb_no_fp32(
        x: torch.Tensor,
        freqs_cis: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        use_real: bool = True,
        use_real_unbind_dim: int = -1,
        sequence_dim: int = 2,
    ) -> torch.Tensor:
        if not use_real:
            # Keep original fallback behavior for non-real path.
            return original_apply_rotary_emb(
                x,
                freqs_cis,
                use_real=use_real,
                use_real_unbind_dim=use_real_unbind_dim,
                sequence_dim=sequence_dim,
            )

        cos, sin = freqs_cis
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        dtype = x.dtype
        cos = cos.to(device=x.device, dtype=dtype)
        sin = sin.to(device=x.device, dtype=dtype)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        return x * cos + x_rotated * sin

    tflux2.apply_rotary_emb = _apply_rotary_emb_no_fp32
    logger.info("rope patch applied: diffusers.transformer_flux2.apply_rotary_emb")
    return True


def maybe_load_scales_from_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    Extract {module_name.input_scale, module_name.weight_scale} from a loaded state_dict.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".input_scale") or k.endswith(".weight_scale"):
            out[_normalize_scale_key(k)] = v
    return out


def validate_fp8_readiness(transformer: nn.Module, scales: dict[str, torch.Tensor]) -> dict[str, Any]:
    """
    Basic production validation for FP8 mapping completeness.
    """
    linear_modules = _collect_linear_module_names(transformer)
    missing_input = [n for n in linear_modules if f"{n}.input_scale" not in scales]
    missing_weight = [n for n in linear_modules if f"{n}.weight_scale" not in scales]
    report = {
        "total_linear_modules": len(linear_modules),
        "missing_input_scale": len(missing_input),
        "missing_weight_scale": len(missing_weight),
        "missing_input_scale_modules": missing_input[:32],
        "missing_weight_scale_modules": missing_weight[:32],
        "fp8_supported_gpu": bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        "triton_available": _is_triton_available(),
    }
    logger.info("fp8 readiness report: %s", report)
    return report


def _is_triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def _resolve_state_dict_tensor(
    state_dict: dict[str, Any],
    module_name: str,
    suffix: str,
) -> Any | None:
    candidates = (
        f"{module_name}.{suffix}",
        f"transformer.{module_name}.{suffix}",
        f"model.{module_name}.{suffix}",
    )
    for key in candidates:
        if key in state_dict:
            return state_dict[key]
    return None


def _resolve_act_scale(act_scales: dict[str, Any], module_name: str) -> torch.Tensor | None:
    candidates = (
        module_name,
        f"{module_name}.input_scale",
        f"transformer.{module_name}",
        f"transformer.{module_name}.input_scale",
    )
    for key in candidates:
        if key in act_scales:
            value = act_scales[key]
            if isinstance(value, torch.Tensor):
                return value.float().reshape(())
            return torch.tensor(float(value), dtype=torch.float32)
    return None


def _extract_weight_scale_from_qtensor(qweight: Any) -> torch.Tensor | None:
    # Best-effort extraction for common torchao tensor layouts.
    for attr in ("scale", "scales", "_scale", "_scales"):
        if hasattr(qweight, attr):
            value = getattr(qweight, attr)
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.float().reshape(())
                return value.float().mean().reshape(())
            try:
                return torch.tensor(float(value), dtype=torch.float32)
            except Exception:
                pass
    if hasattr(qweight, "tensor_impl"):
        impl = getattr(qweight, "tensor_impl")
        for attr in ("scale", "scales"):
            if hasattr(impl, attr):
                value = getattr(impl, attr)
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        return value.float().reshape(())
                    return value.float().mean().reshape(())
                try:
                    return torch.tensor(float(value), dtype=torch.float32)
                except Exception:
                    pass
    return None


def _dequantize_to_fp8_tensor(qweight: Any, target_device: torch.device) -> torch.Tensor:
    if isinstance(qweight, torch.Tensor):
        return qweight.to(device=target_device, dtype=torch.float8_e4m3fn)
    if hasattr(qweight, "dequantize"):
        dense = qweight.dequantize()
        if not isinstance(dense, torch.Tensor):
            raise RuntimeError("quantized weight dequantize() did not return a torch.Tensor")
        return dense.to(device=target_device, dtype=torch.float8_e4m3fn)
    raise RuntimeError(f"unsupported quantized weight type: {type(qweight)}")


def load_torchao_fp8_static_model(
    *,
    ckpt_path: str,
    base_model_or_factory: Callable[[], nn.Module] | nn.Module,
    device: str | torch.device = "cuda",
    quantize_backend: str = "compile",
    require_full_coverage: bool = True,
    strict_checkpoint: bool = True,
    verbose_print: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load a torchao static-FP8 checkpoint (e.g. photoroom format) into a Flux2 transformer.

    Expected checkpoint structure:
    - state_dict: quantized weights
    - act_scales: per-linear static activation scales
    - fp8_dtype: optional string (e.g. "float8_e4m3fn")
    """
    logger.info("loading torchao fp8 static checkpoint: %s", ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"torchao checkpoint must be a dict, got: {type(payload)}")
    if "state_dict" not in payload or "act_scales" not in payload:
        raise RuntimeError("torchao checkpoint must contain `state_dict` and `act_scales`")

    q_state_dict = payload["state_dict"]
    act_scales = payload["act_scales"]
    fp8_dtype = payload.get("fp8_dtype", "unknown")
    if not isinstance(q_state_dict, dict) or not isinstance(act_scales, dict):
        raise RuntimeError("invalid torchao checkpoint: `state_dict`/`act_scales` must be dicts")

    model = base_model_or_factory() if callable(base_model_or_factory) else base_model_or_factory
    model = model.to(device)

    linear_modules: list[tuple[str, nn.Linear]] = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    logger.info(
        "torchao loader: fp8_dtype=%s linear_modules=%d q_state_tensors=%d act_scales=%d",
        fp8_dtype,
        len(linear_modules),
        len(q_state_dict),
        len(act_scales),
    )
    if verbose_print:
        print(
            "[torchao-loader] fp8_dtype=",
            fp8_dtype,
            " linear_modules=",
            len(linear_modules),
            " q_state_tensors=",
            len(q_state_dict),
            " act_scales=",
            len(act_scales),
        )

    scales: dict[str, torch.Tensor] = {}
    loaded_weights = 0
    missing_weight_keys: list[str] = []
    missing_act_scale: list[str] = []
    weight_scale_fallback: list[str] = []
    layer_debug_sample: list[dict[str, Any]] = []
    sample_limit = 24

    for module_name, linear in linear_modules:
        qweight = _resolve_state_dict_tensor(q_state_dict, module_name, "weight")
        if qweight is None:
            missing_weight_keys.append(module_name)
            logger.warning("torchao loader missing quantized weight for module=%s", module_name)
            continue

        try:
            fp8_weight = _dequantize_to_fp8_tensor(qweight, target_device=linear.weight.device)
            if fp8_weight.shape != linear.weight.shape:
                raise RuntimeError(
                    f"shape mismatch for {module_name}: ckpt={tuple(fp8_weight.shape)} model={tuple(linear.weight.shape)}"
                )
            with torch.no_grad():
                linear.weight.copy_(fp8_weight)
        except Exception as exc:
            logger.exception("torchao weight load failure for module=%s", module_name)
            raise RuntimeError(f"failed loading quantized weight for module `{module_name}`: {exc}") from exc

        if linear.bias is not None:
            qb = _resolve_state_dict_tensor(q_state_dict, module_name, "bias")
            if qb is not None and isinstance(qb, torch.Tensor):
                with torch.no_grad():
                    linear.bias.copy_(qb.to(device=linear.bias.device, dtype=linear.bias.dtype))

        in_scale = _resolve_act_scale(act_scales, module_name)
        if in_scale is None:
            missing_act_scale.append(module_name)
            in_scale = torch.tensor(1.0, dtype=torch.float32)
            logger.warning("torchao loader missing act scale for module=%s; fallback input_scale=1.0", module_name)
        w_scale = _extract_weight_scale_from_qtensor(qweight)
        if w_scale is None:
            # Conservative fallback: if no explicit quant scale is discoverable, keep neutral multiplier.
            w_scale = torch.tensor(1.0, dtype=torch.float32)
            weight_scale_fallback.append(module_name)
            logger.warning("no weight scale found for %s; fallback weight_scale=1.0", module_name)

        scales[f"{module_name}.input_scale"] = in_scale
        scales[f"{module_name}.weight_scale"] = w_scale
        loaded_weights += 1

        if len(layer_debug_sample) < sample_limit:
            layer_debug_sample.append(
                {
                    "module": module_name,
                    "qweight_type": type(qweight).__name__,
                    "loaded_weight_shape": tuple(linear.weight.shape),
                    "input_scale": float(in_scale.item()),
                    "weight_scale": float(w_scale.item()),
                }
            )

    if strict_checkpoint and missing_weight_keys:
        raise RuntimeError(
            f"torchao checkpoint missing {len(missing_weight_keys)} linear weights; sample={missing_weight_keys[:16]}"
        )
    if missing_act_scale:
        logger.warning("torchao checkpoint missing act scales for %d linears; sample=%s", len(missing_act_scale), missing_act_scale[:16])

    apply_result = apply_fp8_linears(
        model,
        scales=scales,
        allow_cast_non_fp8_weights=False,
        require_full_coverage=require_full_coverage,
        quantize_backend=quantize_backend,
    )
    report = {
        "fp8_dtype": fp8_dtype,
        "linear_modules": len(linear_modules),
        "loaded_weights": loaded_weights,
        "missing_weight_keys": len(missing_weight_keys),
        "missing_act_scale": len(missing_act_scale),
        "weight_scale_fallback": len(weight_scale_fallback),
        "missing_weight_keys_sample": missing_weight_keys[:16],
        "missing_act_scale_sample": missing_act_scale[:16],
        "weight_scale_fallback_sample": weight_scale_fallback[:16],
        "layer_debug_sample": layer_debug_sample,
        "apply_result": {
            "total_linear": apply_result.total_linear,
            "replaced": apply_result.replaced,
            "skipped_no_scale": apply_result.skipped_no_scale,
            "skipped_dtype": apply_result.skipped_dtype,
        },
    }
    logger.info("torchao fp8 static load report: %s", report)
    if verbose_print:
        print("[torchao-loader] load_report:", report)
    return model, report
