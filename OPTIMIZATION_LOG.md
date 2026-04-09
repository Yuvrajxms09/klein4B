# Klein4B Optimization Log

## Goal

Reduce img2img denoising time on NVIDIA GPUs without reducing steps or resolution.

## What we tried

### 1. Full `klein-cuda-c` backend path

We wired the Python pipeline to allow a native backend handoff:

- `cache_dit_klein.py`
  - added `build_klein_c_full_backend(...)`
  - added `enable_klein_c_full_backend(...)`
- `klein_pipeline.py`
  - added `_run_full_backend_img2img(...)`
  - added an early-exit img2img fast path when a full backend is attached

#### Result

This path was logically valid, but on the NVIDIA A100 benchmark it was much slower than the Python pipeline:

- around `110 s` total denoising for `4` steps in the CLI benchmark

#### Conclusion

This is not the fast path for the current NVIDIA setup.

### 2. CUDA kernel microbenchmarks

We benchmarked isolated CUDA ops in Modal with a CUDA devel image.

Custom ops:

- `silu_mul_`
- `adaln_norm`
- `qk_rms_norm_`
- `rope_2d_offset_`

#### Results

The custom kernels were faster than the pure PyTorch references:

- `silu_mul_`: modest win
- `adaln_norm`: strong win
- `qk_rms_norm_`: modest win
- `rope_2d_offset_`: strong win

#### Conclusion

The kernels are real and useful, but standalone kernel wins are not enough to halve DIT by themselves.

### 3. Synthetic fused/block proxies

We benchmarked synthetic fused chains and block-shaped proxies on Modal.

#### Results

The gains were modest:

- fused proxy: about `1.24x`
- resident/reuse proxy: about `1.40x`
- block-style proxy: about `1.07x`

#### Conclusion

Synthetic proxies confirmed the direction, but they did not prove a large end-to-end DIT reduction.

### 4. Real model hook wiring

We patched the actual transformer model source in:

- `/Users/yuvraj/Desktop/realtime/flux2/src/flux2/model.py`

Changes:

- `SiLUActivation.forward`
  - uses `torch.ops.klein_cuda.silu_mul_` on CUDA `float32`
- `QKNorm.forward`
  - uses `torch.ops.klein_cuda.qk_rms_norm_` on CUDA `float32`
- `apply_rope(...)`
  - uses `torch.ops.klein_cuda.rope_2d_offset_` for the real model-shaped RoPE layout when the shapes line up

#### Modal result

We added a model-shaped proxy benchmark and measured:

- `model_hook_ref_ms = 0.7797`
- `model_hook_ker_ms = 0.4549`

This is about a `1.71x` speedup for that proxy.

#### Conclusion

This is the most meaningful win so far, and it proves the real hooks are helping in the actual model-shaped code path.

### 5. Deeper block refactor in `DoubleStreamBlock` and `SingleStreamBlock`

We reduced extra block overhead in:

- qkv splitting
- modulation arithmetic
- MLP gating
- temporary tensor materialization

Changes in `/Users/yuvraj/Desktop/realtime/flux2/src/flux2/model.py`:

- replaced `einops.rearrange` qkv splitting with a local view/permute helper
- used CUDA-friendly in-place style modulation math where safe
- kept the SiLU gate on the custom CUDA path inside the block MLP stream

#### Status

This is implemented locally and syntax-checked, but it still needs a fresh Modal benchmark to quantify the effect on the real model-shaped path.

## What did not work

- Treating the full `klein-cuda-c` backend as the speed path on NVIDIA
- Assuming isolated kernel wins would automatically halve DIT
- Relying on synthetic proxies as a final signal

## What still looks promising

- Keeping the real CUDA ops in the model source
- Removing more Python orchestration inside `DoubleStreamBlock` and `SingleStreamBlock`
- Reducing intermediate tensor materialization
- Fusing more of the block flow if the data layout allows it

## C/CUDA ideas still worth copying

These showed up as recurring wins in the reference C folders:

- preallocated work buffers for attention and MLP intermediates
- fused QKV + MLP input projections where layout allows it
- fused output projection + concat/gated add patterns
- cached GPU weights across the whole denoising step
- flash-style attention kernels that avoid extra transposes
- direct GPU RoPE and QK norm paths

## Implemented in the Python model path

These are now reflected in `/Users/yuvraj/Desktop/realtime/flux2/src/flux2/model.py`:

- QK norm routed to the custom CUDA op when possible
- RoPE routed to the custom CUDA op when the layout matches
- SiLU gate routed to the custom CUDA op
- explicit preference for flash/memory-efficient SDPA backends

## Notes on cached weights

In this Python model, the practical equivalent of cached GPU weights is:

- load the model once
- move it to CUDA once
- keep it resident for the entire denoising run

We are not adding a separate manual weight cache layer in Python unless a future benchmark shows a real gain.

## Removed as non-winning

- buffer-backed concat for the block joins
- standalone concat reuse benchmark

These are still the main ideas worth continuing to port into the real model path.

## What to avoid for now

- More work on the current full backend as the main NVIDIA speed path
- More benchmark claims without measured GPU results
- Broad rewrites outside the actual hot block path

## Next steps

1. Push further into `DoubleStreamBlock` and `SingleStreamBlock`.
2. Reduce extra reshapes/copies around QKV, RoPE, and attention.
3. Re-benchmark the actual model path on Modal.
4. Keep only changes that show real speedup.
