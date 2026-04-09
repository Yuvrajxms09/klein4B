# CUDA denoiser porting plan (from `klein-cuda-c`)

This repository now supports replacing only transformer denoising calls via a custom CUDA op hook.

## Current architecture

- `Flux2KleinPipeline.__call__` runs denoising through `_run_denoiser(...)`.
- Default path stays on `self.transformer(...)`.
- Optional path uses a registered callback (`set_cuda_denoiser`) and can be connected to `torch.ops` via:
  - `enable_cuda_denoiser_op(pipe, "klein_cuda.denoise_step", ...)`.

Only denoising is swapped. Scheduler, text encoder, VAE, and preprocessing remain unchanged.

## Option A bridge in this repo

`klein_c_bridge/` now provides a strict runtime bridge to `klein-cuda-c`:

- Loads `flux_ctx` from model dir.
- Delegates denoising to `flux_transformer_forward_with_refs` / `flux_transformer_forward_with_multi_refs` through bridge C API.
- Integrates into `Flux2KleinPipeline` through `set_cuda_denoiser`.

This follows `klein-cuda-c` backend semantics for denoising compute path, including multi-reference flow.

For same-path execution across denoising + sampling, use `klein_c_full_backend.py`:
- `KleinCFullBackend.img2img(...)` delegates to `flux_img2img`
- `KleinCFullBackend.multiref(...)` delegates to `flux_multiref`

## Ported in this repo (kernel stage)

Under `cuda_kernels/`, these CUDA ops are now ported and registered as `torch.ops.klein_cuda`:

- `silu_mul_`
- `adaln_norm`
- `qk_rms_norm_`
- `rope_2d_offset_`

These are direct low-level kernels for transformer internals and are ready to be used from a custom backend path.

## What still needs to be ported

`klein-cuda-c` speedups come from full block-level CUDA execution, not a single matmul op.
To reproduce that in this repo, the custom op must include:

1. **Block execution paths**
   - `double_block_forward_cuda`
   - `single_block_forward_cuda_chained`
2. **CUDA kernels / fused ops**
   - RMSNorm (standalone, where needed)
   - softmax-attention and causal softmax
   - fused SiLU/SwiGLU and gated residual ops
3. **Memory strategy**
   - persistent weight cache on GPU
   - activation tensor pool
   - minimal host/device transfers in denoising loop

## Integration constraints in this codebase

- Keep output numerically aligned with current transformer semantics.
- Preserve shape contract: denoiser returns `[B, seq, C]`.
- For fixed run profile (`576x384`, 4 steps, cfg=1), use:
  - `enforce_cfg1=True`
  - `expected_hidden_tokens` guard to catch shape drift.

## Validation checklist

1. **Correctness**
   - Compare custom denoiser output vs baseline transformer output per-step.
   - Compare final image drift over full 4-step run.
2. **Performance**
   - Track `transformer_ms` from pipeline profile before/after.
   - Confirm warm cache behavior (weights resident, no repeated upload).
3. **Stability**
   - Ensure fallback path still works when custom op is absent.
