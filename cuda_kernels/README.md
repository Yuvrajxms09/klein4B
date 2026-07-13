# CUDA kernels for Klein transformer ops

This package ports selected specialized CUDA kernels from `../klein-cuda-c/flux_cuda.cu`
into a PyTorch custom-op extension.

Current ops (namespace: `torch.ops.klein_cuda`):

- `silu_mul_(gate, up)` -> in-place fused SiLU(gate) * up
- `adaln_norm(x, shift, scale, eps)` -> LayerNorm-style AdaLN modulation
- `qk_rms_norm_(q, k, qw, kw, eps)` -> in-place per-head Q/K RMSNorm
- `rope_2d_offset_(x, cos, sin, seq_offset, seq_len)` -> in-place RoPE on `x[seq_offset:seq_offset+seq_len]`
- `fused_qkv_rope_qk_norm_(qkv, qw, kw, cos, sin, seq_offset, seq_len)` -> BF16/FP32 QKV Q/K RMSNorm plus 2D RoPE, returning views

Notes:

- These are low-level kernels intended for transformer block integration.
- `denoise_step` is not implemented here yet; this package is the kernel porting stage.
- The operators register Meta implementations so TorchDynamo/Inductor can trace
  the compiled transformer without treating the CUDA calls as graph-hostile
  Python fallbacks.
- Operators do not allocate native CUDA workspaces. Any temporary tensors are
  owned by PyTorch, which preserves CUDA Graph compatibility. Persistent block
  buffers are managed by the transformer patch layer and must be warmed before
  latency measurement.
- The fused QKV operator currently requires `seq_offset == 0`, which is the
  layout used by the Klein transformer patch. Unsupported layouts fall back to
  the Diffusers attention path and are logged once.

## Build

```bash
python3 setup.py build_ext --inplace
```

## Quick smoke test

```python
import torch
import klein_cuda_ext

gate = torch.randn(4, 8, device="cuda", dtype=torch.float32)
up = torch.randn(4, 8, device="cuda", dtype=torch.float32)
torch.ops.klein_cuda.silu_mul_(gate, up)
```

## Qwen-style Triton QKV packing

`triton_qkv.py` ports the useful block-boundary pattern from the local
`qwen-image-optimizations` repository without adopting its FP8 checkpoint or
attention backend. For each Klein double-stream block it consumes the two fused
TorchAO NVFP4 QKV GEMM outputs and performs the following in one graph-visible
Triton launch:

```text
QKV split -> Q/K RMSNorm -> exact RoPE -> joint text/image Q/K/V packing
```

For each single-stream block, a second operator reads Q/K/V directly from the
contiguous fused QKV+MLP projection output and writes attention-ready Q/K/V.
After exact attention, a third operator flattens the attention heads, computes
SwiGLU directly from the projection suffix, and writes the concatenated input
for the output NVFP4 GEMM. This replaces separate flatten/SwiGLU/concatenation
work with one coalesced Triton launch while preserving Klein's parallel
attention/MLP architecture.

For Klein's 3072 attention and 9216 MLP widths, the post-attention kernel pairs
the 12 attention tiles with the first 12 of 36 MLP tiles. This reduces the grid
from 48 to 36 programs per token row and avoids mixed attention/MLP addressing
inside each output tile.

The operator returns PyTorch-owned tensors, uses `torch.library.triton_op`, and
therefore remains visible to `torch.compile` and CUDA Graphs. Installation is
strict: all five double blocks must already have BF16-first fused projections,
all 20 single blocks must match the fused QKV+MLP contract, and startup
numerical checks must pass before their processors are replaced.
The selected Diffusers attention backend is left unchanged.

## Native SM120 NVFP4 GEMM

`sm120_nvfp4.cpp`, `sm120_nvfp4.cu`, and `sm120_nvfp4.py` provide an optional
CUTLASS 4.5.2 backend for the existing TorchAO W4A4 representation. It preserves
MSLK activation quantization and TorchAO's global/block scale semantics, then:

- borrows TorchAO's packed weight and blocked scale buffers without copying or
  converting them;
- benchmarks six CUTLASS 4.5.2 SM120 NVFP4 MMA tiles at each real Klein shape:
  128x128, 128x64, and 128x32 in N, each with K=128 and K=256;
- fuses dynamic output scaling into CUTLASS's predefined source-free
  `ScaledAcc` epilogue; the current Klein transformer is bias-free, and any
  future biased linear remains on TorchAO;
- retains native dispatch only when the fastest parity-valid tile has at least
  a 5% lower five-round CUDA-event median than the original TorchAO linear;
- uses a compile-safe mutation-only custom-op schema, caller-owned output, no
  GEMM workspace, and a Meta implementation;
- runs a full-graph Inductor/CUDA-graph probe on the first selected production shape before the
  complete transformer is compiled;
- rejects the native backend if the warmed transformer trace loses CUDA Graph
  replay or performs a hot-path CUDA allocation.

Setup reports `kernel_variants`, `shape_backend_choices`, and
`shape_kernel_variants`. Runtime profiling reports native and retained TorchAO
GEMM counts separately. A shape rejection is expected behavior, not a silent
fallback.

The one-shot benchmark treats BF16-first QKV fusion as a transactional
optimization. Its numerical gate compares against separately quantized TorchAO
NVFP4 projections. A rejected fusion restores the ordinary unfused NVFP4
topology before native shape selection, so the SM120 experiment still runs
without weakening the quality gate.

The extension build exposes both CUTLASS's core `include` tree and its
`tools/util/include` tree. The latter contains the official
`cutlass/util/packed_stride.hpp` helper used by the device adapter.

## Attention selection

`flashinfer_attention.py` exposes FlashInfer single-prefill attention through a
fake-tensor-aware custom operator. `attention_backend_request="auto"`
benchmarks native SDPA, FlashAttention 3 when available, and FlashInfer at the
production `[1, 2240, 24, 128]` shape. Exact profiles consider only exact
backends. The quality-tradeoff profiles additionally benchmark FlashInfer's
FP16-QK-reduction path and, when exposed by the installed FlashInfer build, its
SM120 NVFP4 attention path. The latter
includes BF16-to-FP4 Q/K/V quantization, correction, and both layout conversions
in its measured cost. A non-native backend is selected only when it passes its
declared numerical gate, is at least 5% faster, and survives full-graph Inductor
compilation. The warmed trace must still show the selected attention kernels and
CUDA Graph replay when graph capture is enabled; otherwise the benchmark rejects
the candidate.

## Four-step quality-tradeoff profile

`--optimization-profile one-shot-fast` preserves 576x384 output, four scheduler
updates, prompt encoding, TAEF2 encode/decode, and the optimized kernel stack. It
uses Cache-DiT's public static SCM policy with mask `1110`, so the final
transformer result is reused instead of executing a fourth full 25-block
forward. This is an explicit quality/compute tradeoff, not an exact four-forward
benchmark. The post-warmup trace must show exactly 276 NVFP4 GEMMs and 75
attention invocations before timed requests. Approximate attention is considered
but is selected only through the measured numerical/performance gate above.

`--optimization-profile one-shot-aggressive` uses static SCM mask `1100` and
requires exactly the architecture-derived work pattern: 196 NVFP4 GEMMs and 50
attention invocations, versus 356 and 100 in the full fused topology. It still
executes four scheduler updates, but only the first two contain full transformer
forwards. This is the profile with a plausible path to roughly 80 ms; its larger
quality tradeoff is made explicit and its 20 measured outputs are saved for
review.

`--optimization-profile one-shot-80` is the direct latency-target profile. It
keeps scheduler steps 0, 1, 2, and 3, computes complete transformer predictions
for steps 0 and 1, then uses first-order linear prediction extrapolation for
steps 2 and 3. Unlike Cache-DiT, it keeps ordinary transformer CUDA Graph replay
and executes no top-level transformer linears on reused steps. The warmed trace
must contain exactly 178 NVFP4 GEMMs, 50 attention invocations, 10 double-block
packing kernels, and 40 single-block packing/output kernels. This profile has
the largest explicit quality tradeoff and exists specifically for the ~80 ms
E2E target.
