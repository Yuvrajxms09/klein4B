# CUDA kernels for Klein transformer ops

This package ports selected specialized CUDA kernels from `../klein-cuda-c/flux_cuda.cu`
into a PyTorch custom-op extension.

Current ops (namespace: `torch.ops.klein_cuda`):

- `silu_mul_(gate, up)` -> in-place fused SiLU(gate) * up
- `adaln_norm(x, shift, scale, eps)` -> LayerNorm-style AdaLN modulation
- `qk_rms_norm_(q, k, qw, kw, eps)` -> in-place per-head Q/K RMSNorm
- `rope_2d_offset_(x, cos, sin, seq_offset, seq_len)` -> in-place RoPE on `x[seq_offset:seq_offset+seq_len]`

Notes:

- These are low-level kernels intended for transformer block integration.
- `denoise_step` is not implemented here yet; this package is the kernel porting stage.

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
