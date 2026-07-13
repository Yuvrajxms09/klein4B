from __future__ import annotations

from typing import Any

import torch


_extension: Any | None = None
KERNEL_VARIANTS = {
    0: "128x128x256",
    1: "128x128x128",
    2: "128x64x256",
    3: "128x64x128",
    4: "128x32x256",
    5: "128x32x128",
}


def register_extension(extension: Any) -> None:
    global _extension
    _extension = extension


def prepare_linear(
    module: torch.nn.Linear, weight: Any, *, max_rows: int = 4096
) -> None:
    if _extension is None:
        raise RuntimeError("SM120 NVFP4 extension has not been registered")
    n, k = map(int, weight.shape)
    if max_rows <= 0 or n % 2 or k % 64:
        raise ValueError(
            f"unsupported SM120 NVFP4 shape: max_rows={max_rows}, N={n}, K={k}"
        )
    if module.bias is not None:
        raise ValueError(
            "SM120 NVFP4 native dispatch currently supports bias-free Klein linears only"
        )

    representative_rows = (1, 128, 512, 1216, 1728, 2304, 2816, max_rows)
    workspace_sizes = {
        int(_extension.workspace_size(rows, n, k, variant))
        for rows in representative_rows
        for variant in KERNEL_VARIANTS
    }
    if workspace_sizes != {0}:
        raise RuntimeError(
            f"SM120 kernels unexpectedly require workspace: {sorted(workspace_sizes)}"
        )

    module.register_buffer(
        "_klein_sm120_weight_qdata_b", weight.qdata, persistent=False
    )
    module._klein_sm120_max_rows = int(max_rows)
    module._klein_sm120_kernel_variant = 0


def release_linear(module: torch.nn.Linear) -> None:
    for name in ("_klein_sm120_weight_qdata_b",):
        module._buffers.pop(name, None)
    for name in ("_klein_sm120_max_rows", "_klein_sm120_kernel_variant"):
        if hasattr(module, name):
            delattr(module, name)


def linear_forward(module: torch.nn.Linear, input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.dtype != torch.bfloat16:
        raise TypeError(f"SM120 NVFP4 requires BF16 input, got {input_tensor.dtype}")
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    if input_2d.shape[0] > module._klein_sm120_max_rows:
        raise RuntimeError(
            f"SM120 NVFP4 row count {input_2d.shape[0]} exceeds prepared maximum "
            f"{module._klein_sm120_max_rows}"
        )

    if module._klein_nvfp4_dynamic_activation_scale:
        activation_scale = torch.max(torch.abs(input_2d)).to(torch.float32) / 2688.0
    else:
        activation_scale = module._klein_nvfp4_activation_scale
    if activation_scale is None or module._klein_nvfp4_weight_scale is None:
        raise RuntimeError("SM120 NVFP4 requires two-level global scaling")

    activation_block_scales, activation_qdata = torch.ops.ao.mslk_quantize_nvfp4(
        input_2d,
        activation_scale.reciprocal(),
    )
    output_scale = (activation_scale * module._klein_nvfp4_weight_scale).to(
        torch.float32
    )
    output = torch.empty(
        (input_2d.shape[0], module.out_features),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.ops.klein_sm120.nvfp4_gemm_out(
        activation_qdata,
        activation_block_scales,
        module._klein_sm120_weight_qdata_b,
        module._klein_nvfp4_weight_block_scales,
        output_scale,
        output,
        module._klein_sm120_kernel_variant,
    )
    return output.reshape(*input_tensor.shape[:-1], module.out_features)
