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

### 6. Latent unpack sync removal

We removed the latent-shape inference sync in:

- `klein_pipeline.py`

Change:

- `Flux2KleinPipeline._unpack_latents_with_ids(...)`
  - now receives the known latent `height` and `width` from the caller
  - no longer falls back to `torch.max(h_ids)` / `torch.max(w_ids)` during unpack

#### Result

The profiler showed about `13.2 ms` of DtoH synchronization in the original trace from that unpack path.

#### Conclusion

This is a real end-to-end win for the same fixed resolution and step count, and it is the cleanest measured latency reduction so far.

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
# 2026-07-11: graph-safe CUDA operator integration

The existing `cuda_kernels` extension was tightened for the 128 ms TorchAO
NVFP4 path:

- Removed per-call `fprintf` diagnostics from CUDA operators; these were
  host-side work on every block invocation and were never suitable for timing.
- Added Meta implementations for every registered `klein_cuda` operator so
  TorchDynamo/Inductor can trace the custom CUDA calls with FakeTensor instead
  of treating them as unsupported Python boundaries.
- Kept all temporary storage PyTorch-owned; the extension does not introduce
  native CUDA allocations or graph-hostile workspaces.
- Left attention backend selection unchanged. FA3 remains available through
  the existing automatic selection, but this optimization does not force it.

This change is intentionally not reported as a measured latency improvement
until it is rebuilt and benchmarked on the RTX PRO 6000 at 384x576 with text
embedding, TAEF2, four denoising steps, and the existing unpack fix included.

# 2026-07-11: device-state and fused QKV preparation

The 128 ms trace identifies repeated device/layout overhead around the
transformer rather than a missing NVFP4 GEMM implementation. The Klein path
now applies the following local changes:

- Cache Diffusers' `_execution_device` only when no offload hook is present;
  offloaded pipelines retain dynamic device resolution.
- Cache fixed-resolution latent IDs on CUDA.
- Keep image-conditioning tokens and image/latent position IDs in persistent
  PyTorch buffers for a request. Only the changing latent prefix is copied on
  subsequent denoising steps.
- Use the direct single-reference packing path instead of a
  squeeze/concatenate/unsqueeze sequence.
- Change the local CUDA extension to launch on PyTorch's current CUDA stream,
  not the default stream. This is required for correct ordering with compiled
  execution and CUDA Graphs.
- Replace the old QKV helper's contiguous Q/K temporaries plus separate norm
  and RoPE launches with one BF16/FP32 fused QKV QK-norm+RoPE kernel. The
  attention computation remains PyTorch Flash/SDPA; the custom naive attention
  kernel is not selected by this change.

The benchmark's cache mode now disables CUDA Graphs because Cache-DiT retains
intermediate tensors across steps, and it captures one non-timed trace that
rejects the configuration unless the NVFP4 GEMM count is below the 436-call
full-compute baseline.

No latency claim is made until the CUDA extension is rebuilt on the target
GPU. The local host has no `nvcc`, so only Python syntax and patch validation
are possible here.
### 7. Current CUDA patch audit

The current opt-in CUDA path keeps TorchAO NVFP4 linears and the compiled
transformer unchanged. It adds only:

- persistent PyTorch-owned denoiser input and ID buffers;
- a BF16/FP32 fused QKV QK-RMSNorm plus 2D RoPE kernel on the current CUDA
  stream;
- Meta registrations so the custom operators remain visible to Dynamo/Inductor;
- one-time logs for extension loading, transformer patch activation, buffer
  allocation, and custom-op fallback.

The fused kernel does not replace attention or NVFP4 GEMMs. The extension is
not enabled by default, and no end-to-end speedup is claimed until a CUDA 13+
SM120 build verifies graph capture, four denoising steps, NVFP4 GEMM count,
and output parity.

### 8. Direct TorchAO NVFP4 dispatch

The optimized TorchAO path now unwraps each `NVFP4Tensor` once during setup and
registers its packed FP4 bytes, blocked E4M3 scales, and tensor scale as
non-persistent module buffers. The hot `Linear.forward` calls the same MSLK
activation quantizer and `torch._scaled_mm` sequence directly, bypassing the
Python tensor-subclass dispatch layer without changing weight format or GEMM
numerics.

The patch is strict: every NVFP4 candidate must be converted and distinct
linear shapes are compared against TorchAO before compilation. The warmed
profiler trace must contain 436 NVFP4 GEMMs for the unfused control or 356 when
BF16-first double-stream QKV fusion is enabled, with no `nvfp4_linear` or
`_dispatch__torch_dispatch__` events.

### 9. Resident four-step execution

The fixed production path can compile all four transformer forwards and exact
FlowMatch Euler updates as one static module. Image/text RoPE tensors, latent
IDs, image-conditioning tensors, VAE BN constants, and scheduler deltas stay on
device. The ordinary Python loop remains the correctness fallback for CFG,
callbacks, Cache-DiT, stochastic scheduling, or non-matching step counts.

Before latency measurement, the benchmark compares the compiled-loop latent
against the ordinary four-step scheduler loop and rejects the run on parity
failure. A warmed trace also rejects repeated RoPE construction.

These paths are implemented and syntax-checked locally. They are not claimed
as measured speedups until the CUDA 13/SM120 run passes the parity and trace
gates above.

### 10. Qwen-image block-boundary port

The useful part of `qwen-image-optimizations` is its treatment of memory-bound
boundaries around library GEMMs, not its FP8 format, reduced scheduler work, or
selective CFG. Klein now has graph-visible Triton operators that fuse QKV
splitting, Q/K RMSNorm, exact RoPE, and final attention packing in all 5 double
and 20 single blocks. The double-block operator runs after BF16-first QKV fusion
and TorchAO NVFP4 quantization, so the projection GEMMs remain TorchAO
`_scaled_mm` calls. The single-block operator reads Q/K/V from the existing
fused QKV+MLP projection output and leaves its MLP suffix unchanged.

The original trace shows why this is only one rung of the ladder. Across four
steps, the 20 single blocks occupy 80 compiled GPU regions totaling roughly
71.6 ms. A representative region contains a roughly 0.31 ms fused QKV+MLP
input GEMM, 0.16 ms output GEMM, 0.25 ms exact attention kernel, and 0.075 ms
SwiGLU/layout kernel. The new packing operators remove repeated layout/norm/RoPE
work and are required to appear exactly 20 double-block and 80 single-block
times in the warmed trace, but a 2x result still requires reducing the executed
single-block GEMM and attention time.

### 11. Single-block post-attention fusion

The repeated single-block boundary now uses a graph-visible Triton operator for
the exact operation below:

```text
attention [B,S,H,D] + fused projection MLP suffix
  -> flatten heads + SwiGLU + output-linear packing
```

This removes the standalone SwiGLU output and `torch.cat` allocation before the
second TorchAO NVFP4 linear. It does not change attention, projection weights,
or residual math. Startup validation compares the fused BF16 output against
`Flux2SwiGLU` plus `torch.cat`; the warmed trace must contain exactly 80 fused
post-attention launches for four steps and no CUDA allocation/free operations.
No speedup is claimed until the target SM120 benchmark passes these gates.

### 12. Direct SM120 cuDNN kernel experiments

The Qwen-style packing experiment was rejected after the target trace showed
13.26 ms of new Triton packing work while aggregate NVFP4 GEMM time changed
from roughly 47.30 ms to 47.04 ms. Whole-loop compilation was also rejected by
latent parity (`cosine ~= 0.986`). Both experiments remain opt-in and are no
longer benchmark defaults.

The next GEMM track keeps TorchAO quantization and `torch.compile`, but replaces
only `_scaled_mm` with FlashInfer's cuDNN `mm_fp4` backend. TorchAO MSLK and
FlashInfer use compatible packed E2M1 data and 128x4 swizzled E4M3 scale-factor
layouts. A `torch.library.custom_op` boundary preserves Dynamo visibility, and
every distinct linear shape is compared against TorchAO before compilation.
The path requires CUDA 13+, cuDNN 9.15+, and is selected explicitly with
`nvfp4_gemm_backend="flashinfer-cudnn"`.

For exact attention, `attention_backend_request="cudnn"` forces PyTorch cuDNN
SDPA while retaining the existing Diffusers processors and Q/K/V tensors. It
is an independent A/B track against the current native `flash_fwd` kernel.
Neither backend is promoted until the warmed profiler proves lower aggregate
kernel time and full latent parity.

#### SM120 result

The combined FlashInfer-cuDNN GEMM and cuDNN-SDPA run was rejected:

- valid TorchAO/native median: 129.89 ms
- FlashInfer-cuDNN plus cuDNN-SDPA median: 132.90 ms
- total GPU kernels: 90.50 ms -> 90.48 ms (no meaningful change)
- reported NVFP4 GEMMs: 47.20 ms -> 45.76 ms
- activation quantization: 0.53 ms -> 5.05 ms
- exact attention: 23.42 ms -> 25.02 ms
- copy operators: 2.10 ms -> 4.45 ms

The small cuDNN GEMM kernel reduction was more than offset by quantization,
attention, and copy overhead. Both cuDNN paths remain opt-in diagnostics and
the default stays TorchAO `_scaled_mm` with native attention.
# One-shot exact profile

`modal_bench_inference_nvfp4.py --optimization-profile one-shot-exact` is the
single combined experiment for the remaining exact, graph-compatible paths. It
enables:

- TorchAO NVFP4 W4A4 for the transformer and the first 27 Qwen decoder layers.
- Qwen base-model execution without the unused LM head or decoder layers 28-36.
- BF16-first transformer QKV fusion before NVFP4 conversion.
- Graph-safe Triton QKV layout, QK RMSNorm/RoPE, and single-block
  attention/SwiGLU output packing.
- Direct raw-tensor TorchAO NVFP4 dispatch, automatic exact attention,
  transformer/text/TAEF2 compilation, and final PIL output.

The profile now also evaluates a native CUTLASS 4.5.2 SM120 NVFP4 backend. It
reuses TorchAO quantization and packed weights directly, and fuses dynamic
global scaling through CUTLASS's predefined source-free `ScaledAcc` epilogue.
The Klein transformer constructs its linears with `bias=False`; a future biased
linear is explicitly retained on TorchAO. Setup benchmarks CUTLASS 4.5.2 SM120
block-scaled MMA tiles 128x128, 128x64, and 128x32 in N, each with K=128 and
K=256, for every unique production shape. The narrow-N families remain measured
candidates rather than defaults. The fastest parity-valid native
tile is promoted only when its five-round CUDA-event median is at least 5%
faster than TorchAO. Rejected
shapes continue on TorchAO. The selected persistent scheduler and compute-only
epilogue require zero workspace; setup verifies this for every supported tile
and production row count.

The custom operator mutates caller-owned output and returns no aliased tensors.
PyTorch functionalization rejects custom operators that return mutable input
aliases, so the seemingly conventional out-op return form is intentionally not
used. The first native candidate must also pass a real full-graph
Inductor/CUDA-graph probe
before native dispatch is allowed for the transformer.
The warmed profiler trace additionally rejects native dispatch if CUDA Graph
replay disappears or any hot-request CUDA allocation is observed.

The SM100 256x256x256, 2x4x1 configuration is deliberately excluded. It is not
an SM120 GeForce kernel configuration. This backend is therefore a measured dispatch
experiment, not an assumption that a larger data-center Blackwell tile applies
to RTX PRO 6000.

The extension uses NVCC `-arch=sm_120f` directly. PyTorch's extension parser
accepts `TORCH_CUDA_ARCH_LIST=12.0` but rejects the family suffix `12.0f`; using
the direct NVCC flag both enables CUTLASS's SM120-family instructions and avoids
an additional plain-SM120 compilation pass.

The sibling `F2K_CUDA` CUTLASS headers are added to the Modal image only when
that local checkout exists. This keeps the ordinary TorchAO benchmark usable
without the experimental native dependency; `one-shot-exact` reports a clear
missing-header error if native compilation is requested without it.

CUTLASS keeps runtime utilities such as `cutlass/util/packed_stride.hpp` under
`tools/util/include`, outside its core `include` tree. The Modal image and JIT
extension therefore expose both official include roots and verify the packed
stride header before invoking NVCC. The SM120 epilogue and mainloop builders
both use CUTLASS's `OpClassBlockScaledTensorOp`, matching the pinned NVIDIA
SM120 NVFP4 example.

The profile intentionally excludes Cache-DiT, the native C block patch,
ModelOpt, FlashInfer-cuDNN GEMM, forced cuDNN attention, and whole-loop compile.
Those paths are either approximate, mutually incompatible, or already rejected
by complete-request timing/parity. The ordinary `baseline` profile remains the
default.

BF16-first QKV fusion is validated against the actual unfused TorchAO NVFP4
runtime, not against BF16. The original projections are intentionally excluded
from the fused model's conversion, so comparing the fused W4A4 result directly
to those BF16 modules incorrectly attributes ordinary NVFP4 quantization error
to fusion. Setup now creates one temporary projection at a time, converts it
with TorchAO's public `quantize_` workflow, and reports both incremental
fused-vs-unfused NVFP4 error and fused-vs-BF16 diagnostic error. If incremental
parity fails, fusion is transactionally rolled back, the original six
projections per double block are quantized normally, Triton fused-QKV packing is
disabled, and the one-shot run continues with 109 linears and 436 transformer
GEMMs. No numerical threshold is relaxed.

CUDA Graph replay does not replay the Python/C++ dispatcher, so profiler traces
can contain every captured GEMM kernel while showing zero
`klein_sm120::nvfp4_gemm_out` CPU events. Native validation therefore counts
the replayed CUTLASS `device_kernel` launches and compares them with the
per-module backend selection table; Torch `_scaled_mm` kernels are counted
separately. The report includes a bounded top-k GPU-kernel histogram when a
count differs. CPU `aten::copy_` spans are also labeled separately from actual
CUDA copy/memcpy events because their asynchronous operator duration is not GPU
copy time.

The first one-shot trace measured the fused single-block post-attention/SwiGLU
packing kernel at 7.68 ms across 80 calls, versus about 6.53 ms for the two old
compiled boundary kernels. The original fused kernel launched 48 programs per
token row and carried attention and MLP address paths in every program. The
revised kernel pairs the 12 attention blocks with the first 12 of 36 MLP blocks,
launches only 36 programs per row, and writes the two output regions directly.
This removes `minimum`, `maximum`, and mixed-result `where` operations while
preserving one launch and the existing numerical validation.

The one-shot profile also overlaps the two independent request inputs: Qwen
prompt encoding runs on one persistent CUDA stream while compiled TAEF2 image
encoding runs on another. Both streams wait on a common request-start event and
the denoiser stream waits for both completion events, so wall and CUDA-event E2E
timing include the slower preparation branch. The pipeline accepts explicitly
precomputed packed image latents and position IDs through a validated public
boundary; the ordinary PIL path is unchanged. Every request still recomputes
both prompt and source-image conditioning. The result reports prompt, VAE, and
overlapped preparation timings independently.

Because native dispatch is selected at exact token counts, `one-shot-exact`
rejects any workload other than 576x384, four steps, guidance 1.0. Supporting
another resolution requires a separate measured shape table rather than reuse
of these selections.

### 13. Dominant-kernel selection and four-step tradeoff profile

Automatic attention selection now benchmarks native SDPA, FlashAttention 3
when importable, and FlashInfer single-prefill at Klein's exact
`[1, 2240, 24, 128]` shape. FlashInfer is represented by a graph-visible custom
operator and can replace native attention only after numerical parity, a 5%
microbenchmark win, full-graph Inductor compilation, and warmed-trace CUDA Graph
validation. The saved benchmark includes the complete selection report.

`one-shot-fast` is a separate quality-tradeoff profile for the fixed 576x384,
four-step request. It keeps four scheduler callbacks but applies Cache-DiT's
public static SCM policy with `steps_mask=1110`, reusing the previous
transformer result at the final scheduler step. This directly removes both
NVFP4 GEMMs and attention calls rather than optimizing secondary copies. Cache
validation runs only after compile warmup, recognizes both Torch and native
CUTLASS kernels, and requires the architecture-derived 276 GEMMs and 75
attention invocations instead of the full fused topology's 356 and 100. This
profile must not be reported as four full transformer forwards.

`one-shot-aggressive` extends the same explicit contract with static mask
`1100`. It requires at least a 44% GEMM reduction. Two full block passes account
for 178 GEMMs, while nine top-level transformer linears still execute on each
cached step, so the architecture-aware expectation is about 196 GEMMs total;
attention should fall from 100 toward 50 invocations.
This is the only current profile whose transformer-work budget makes an ~80 ms
request plausible. It retains four scheduler updates but not four full
transformer evaluations, and therefore requires visual acceptance of the saved
20-prompt output suite.

The quality-tradeoff profiles also feature-detect FlashInfer's SM120 NVFP4
attention implementation. Modal's package mirror currently provides 0.6.14;
when that build does not expose the newer API, the candidate records its import
error and native/exact FlashInfer remains eligible. The benchmark includes Q/K/V layout conversion,
FP4 quantization, the attention kernel, and output layout restoration; it does
not compare only the inner FP4 kernel. Native BF16 attention remains selected
unless the complete candidate is at least 5% faster, has relative L2 at most
0.15 and cosine at least 0.98, and passes a full-graph compile probe. Exact
profiles never consider this candidate.

Static-cache validation is architecture-exact rather than threshold-only. With
the fused 89-linear transformer topology, each full step executes 89 NVFP4
GEMMs and each reused step retains nine top-level linears. Therefore `1100` must
produce exactly `2 * 89 + 2 * 9 = 196` GEMMs and 50 attention invocations. The
benchmark aborts before timing if either count differs, while separately proving
that scheduler callbacks still run for steps 0, 1, 2, and 3.

### 14. Direct four-step prediction reuse

`one-shot-80` removes Cache-DiT from the target path. The pipeline executes the
compiled denoiser for scheduler steps 0 and 1, clones those graph-owned outputs,
and uses first-order extrapolation for steps 2 and 3:

```text
prediction(step 2) = prediction(step 1) + delta
prediction(step 3) = prediction(step 1) + 2 * delta
delta = prediction(step 1) - prediction(step 0)
```

All four scheduler updates, prompt encoding, TAEF2 encode/decode, unpacking, and
PIL conversion remain in E2E timing. The two reused steps skip the complete
transformer, including its nine top-level linears, so the fused topology must
show exactly `2 * 89 = 178` NVFP4 GEMMs and `2 * 25 = 50` attention invocations.
The transformer retains normal `max-autotune` CUDA Graphs rather than Cache-DiT's
no-cudagraph compile mode. This is an intentional approximation and must not be
described as four full transformer evaluations.

### 15. Calibrated static NVFP4 transformer scales

The native SM120 trace showed that the CUTLASS GEMMs themselves were not the
only cost at the linear boundary. Dynamic NVFP4 global scaling still launched
an activation absolute-value pass and reduction for every projection. The
benchmark now exposes an opt-in exact four-forward A/B track with
`--enable-static-transformer-activation-scales`.

This path uses TorchAO's `per_tensor_amax_to_scale` semantics and calibrates all
active fused transformer linears on eight separate 576x384, four-step requests.
It freezes only the global activation scale. MSLK still computes FP8 block
scales and packs FP4 activations dynamically, and all TorchAO packed weights,
exact attention calls, scheduler steps, and model blocks remain unchanged.

Calibration is setup-only and excluded from measured requests. The run aborts
unless every active NVFP4 linear executes during calibration and receives a
finite positive scale. Direct-dispatch setup compares CUTLASS against the
Torch `_scaled_mm` path using the same calibrated scale, while separately
reporting the numerical delta from dynamic scaling. This experiment is not a
promoted default until the saved 20-prompt image suite and full E2E timing beat
the dynamic-scale control.

Before timing, a held-out prompt also runs through both the original dynamic
TorchAO transformer and the calibrated static transformer with identical image,
noise, resolution, and four scheduler steps. Cosine and relative L2 are logged
as diagnostics only; they do not gate execution. The benchmark rejects only
non-finite latents, while quality acceptance is based on the saved 20-image
suite.

#### RTX PRO 6000 result

The first completed static-scale `one-shot-exact` run passed the execution
gates and saved all 20 measured PIL outputs:

- E2E CUDA median: `110.846 ms`
- E2E wall median: approximately `110.87 ms`
- E2E CUDA mean: `111.315 ms`
- E2E CUDA minimum: `110.187 ms`
- E2E CUDA p95: `114.061 ms`
- NVFP4 GEMMs: `356` calls, `47.270 ms`
- exact attention: `100` calls, `23.901 ms`
- total GPU kernels: `90.062 ms`
- static activation scales: `89/89`
- selected GEMM modules: `84` native CUTLASS, `5` Torch `_scaled_mm`
- CUDA Graph launches: `5`
- hot-request CUDA allocations: `0`

Against the previous exact-quality median of `113.034 ms`, this is about a
`2.19 ms` median improvement. The prior minimum was `112.230 ms`, so the
minimum improved by about `2.04 ms`. Static scaling removed the visible global
activation `abs/amax` kernels, but FP4 block packing still costs `2.561 ms` and
the aggregate GEMM time remains effectively unchanged from the earlier
`47.38 ms` trace. QKV/post-attention Triton packing costs about `11.37 ms` in
aggregate, while exact attention remains approximately `24 ms`.

This is a real but small exact-path win. It does not materially alter the
remaining transformer floor. Promotion still depends on visual acceptance of
the saved image suite.
