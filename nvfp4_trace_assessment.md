# NVFP4 Warm-3 Eager Latent Trace Assessment

Trace file:

- [`/Users/yuvraj/Desktop/realtime/klein4B/klein4b_img2img_nvfp4_warm3_eager_latent.json`](./klein4b_img2img_nvfp4_warm3_eager_latent.json)

This assessment is based on the warmup-3 Modal profiler trace for the NVFP4 Klein 4B img2img setup:

- `576x384`
- `4` denoising steps
- `guidance_scale=1.0`
- `output_type=latent`
- `NVFP4` transformer with dynamic activation scaling
- `cache-dit` enabled
- attention backend reported as `native`
- transformer compiled with `max-autotune`
- transformer QKV projections fused before profiling

## Executive Summary

The trace confirms the expected shape of the bottleneck:

- the transformer dominates end-to-end runtime
- the hot path is not just attention math
- the real cost is the compiled transformer block stack: linear layers, attention, dtype/layout churn, and NVFP4 dispatch overhead
- the fused attention kernels are present, but they are not the only or even the main source of time

The practical conclusion is that NVFP4 helped, but the remaining latency is now mostly in:

- transformer block glue
- compiled graph management
- `aten::linear`
- `aten::copy_`, `aten::to`, `aten::_to_copy`
- Triton/compiled attention and norm fusion

## Top-Level Timing

The trace root shows:

- `pipeline_call`: about `128.712 ms`
- `transformer_forward`:
  - user annotation: about `70.224 ms`
  - GPU user annotation: about `116.852 ms`

That means the transformer is still the dominant cost inside the benchmark.

## Main Findings

### 1. Transformer is still the bottleneck

The largest meaningful model-level scope is:

- [`Flux2Transformer2DModel_0`](not a direct file scope, but present in the trace)
- about `70.304 ms` across `4` calls

The compile/runtime wrapper around it also contributes:

- `CachedBlocks_Pattern_3_4_5_0`: still visible, but no longer the dominant visible wrapper cost in the trace
- `CachedBlocks_Pattern_0_1_2_0`: still visible, but secondary to the NVFP4 linear path

This shows the compile path is not a pure kernel replay path yet, but the eager-latent trace shifts the emphasis further toward the transformer math itself and away from compile wrapper overhead.

### 2. Linear layers are still expensive even in NVFP4

The single biggest math primitive visible at the trace level is:

- `aten::linear`: about `32.369 ms` across `36` calls

Associated NVFP4 dispatch work is visible too:

- `torchao/prototype/mx_formats/nvfp4_tensor.py(526): nvfp4_linear`: about `31.970 ms`
- `torchao/prototype/mx_formats/nvfp4_tensor.py(124): to_nvfp4`: about `24.404 ms`
- `torchao/utils.py(667): _dispatch__torch_dispatch__`: about `32.753 ms`

This is the most important observation for the NVFP4 case:

- NVFP4 is not free
- quantization/dequantization and dispatch overhead are now a visible part of the total
- the transformer still spends a lot of time moving through the TorchAo/NVFP4 plumbing

### 3. Attention is fused, but still nontrivial

There are clear fused attention kernels in the trace:

- `pytorch_flash::flash_fwd_kernel...`: about `23.814 ms` across `100` kernel launches
- `triton_poi_fused__scaled_dot_product_flash_attention...`: about `1.282 ms` and `1.060 ms`
- another fused attention variant: about `0.302 ms`

This suggests:

- attention backend is not the main remaining problem
- the attention path is already being fused reasonably well
- but attention still consumes enough time that block-level fusion matters

### 4. CPU-side tensor churn is still significant

The trace shows large counts of layout and conversion operations:

- `aten::copy_`: about `6.962 ms`
- `aten::to`: about `5.654 ms`
- `aten::_to_copy`: about `5.310 ms`
- `aten::reshape`: about `1.310 ms`
- `aten::transpose`: about `0.882 ms`
- `aten::view`: about `0.451 ms`
- `aten::cat`: about `0.573 ms`

These are individually smaller than linear and attention, but they are exactly the kind of residual cost that prevents the model from dropping to a much lower latency tier.

### 5. Compile overhead is still visible

The trace contains a lot of compile/runtime scaffolding:

- `TorchDynamo Cache Lookup`: about `5.157 ms`
- `Torch-Compiled Region` events are still present, but the eager-latent run spends less of its visible time in wrapper bookkeeping than the older compiled-pil trace

The compiled path is helping, but the model is still paying for graph management and wrapper overhead rather than just replaying a stable fused graph.

## Kernel-Level Bottlenecks

### Dominant CUDA kernel

The heaviest CUDA kernel in the trace is:

- `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_relu`
- about `47.487 ms` across `436` launches

This is the key NVFP4-weighted GEMM path.

Interpretation:

- the model is using a CUTLASS-backed low-precision GEMM kernel
- that kernel is clearly one of the main compute sinks
- the benefit of NVFP4 is real, but the path still launches many times

### Attention kernels

Relevant fused attention kernels:

- `pytorch_flash::flash_fwd_kernel<...>`: about `23.814 ms` across `100` launches
- `triton_poi_fused__scaled_dot_product_flash_attention...`: about `1.282 ms`
- `triton_poi_fused__scaled_dot_product_flash_attention...`: about `1.060 ms`

Interpretation:

- attention is already reasonably optimized
- the bigger opportunity is not “replace attention with something approximate”
- the better move is to reduce the number of surrounding ops and improve block fusion

### Norm / modulation fusion

There are multiple fused Triton kernels for norms and modulation:

- `triton_red_fused__fused_rms_norm...`
- `triton_poi_fused___rshift____to_copy...native_layer_norm...`
- `triton_per_fused__to_copy__unsafe_view_abs_add_amax_bitwise_and...silu...`

These show that some of the block math is already fused, but there is still a lot of launch variety.

## Transformer Layer Signals

The trace includes the high-level module names:

- `Flux2TransformerBlock_0` through `Flux2TransformerBlock_4`
- `Flux2SingleTransformerBlock_0` through `Flux2SingleTransformerBlock_19`
- `Flux2Modulation_0`, `_1`, `_2`
- `Flux2TimestepGuidanceEmbeddings_0`
- `Flux2PosEmbed_0`

The important point is not the module names alone, but the fact that:

- there are 5 double blocks
- there are 20 single blocks
- the single-block family is still too expensive when multiplied by step count

## What This Means For Optimization

### Best remaining target

The best remaining target is not the attention backend alone.

It is the transformer block boundary itself:

- `linear -> split -> norm -> rope -> attention -> concat -> linear`
- repeated across 20 single blocks
- plus the 5 double blocks

### Highest-value action

The most valuable optimization would be a tighter fused path that reduces:

- `aten::linear`
- `aten::copy_`
- `aten::to`
- `aten::cat`
- `reshape` / `transpose` churn
- TorchAo dispatch overhead

### What not to chase first

Do not chase approximate attention first.

The trace does not show attention as the dominant issue. The issue is the full block pipeline around it.

## Practical Priority List

1. Reduce NVFP4 dispatch overhead around linear layers.
2. Fuse the single-block path more aggressively.
3. Cut `to` / `copy_` / `cat` churn.
4. Keep attention exact and fused.
5. Reduce compile/runtime wrapper overhead if possible.

## Bottom Line

This trace is consistent with a model that has already moved past the naive bottleneck stage.

NVFP4 has lowered the raw math cost, but the remaining time is now dominated by:

- transformer block execution structure
- low-precision GEMM launch count
- TorchAo/NVFP4 dispatch and conversion overhead
- fused attention plus surrounding glue
- compile/runtime wrapper overhead

If the goal is another large latency drop, the next gain will come from block fusion and tensor-flow simplification, not from swapping attention for an approximate variant.
