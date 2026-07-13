#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <limits>

namespace {
constexpr float LOG2E = 1.4426950408889634074f;
constexpr int BLOCK_NORM = 256;
constexpr int BLOCK_1D = 256;
constexpr int ATTN_THREADS = 128;
constexpr int ATTN_TILE = 32;
}

__global__ void k_silu_mul(float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void k_adaln_norm(
    float* out,
    const float* x,
    const float* shift,
    const float* scale,
    int seq,
    int hid,
    float eps) {
    int row = blockIdx.x;
    if (row >= seq) return;

    const float* xr = x + row * hid;
    float* outr = out + row * hid;

    __shared__ float smean[BLOCK_NORM];
    __shared__ float svar[BLOCK_NORM];
    float sm = 0.0f;
    float sv = 0.0f;

    for (int i = threadIdx.x; i < hid; i += blockDim.x) sm += xr[i];
    smean[threadIdx.x] = sm;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smean[threadIdx.x] += smean[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smean[0] / static_cast<float>(hid);

    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        float d = xr[i] - mean;
        sv += d * d;
    }
    svar[threadIdx.x] = sv;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) svar[threadIdx.x] += svar[threadIdx.x + s];
        __syncthreads();
    }
    float rstd = rsqrtf(svar[0] / static_cast<float>(hid) + eps);

    for (int i = threadIdx.x; i < hid; i += blockDim.x) {
        float norm = (xr[i] - mean) * rstd;
        outr[i] = (1.0f + scale[i]) * norm + shift[i];
    }
}

__global__ void k_qk_rms_norm(
    float* q,
    float* k,
    const float* qw,
    const float* kw,
    int seq,
    int heads,
    int hdim,
    float eps) {
    int idx = blockIdx.x;
    int s = idx / heads;
    int h = idx % heads;
    if (s >= seq) return;

    float* qh = q + s * heads * hdim + h * hdim;
    float* kh = k + s * heads * hdim + h * hdim;

    __shared__ float sq[BLOCK_NORM];
    __shared__ float sk[BLOCK_NORM];
    float sumq = 0.0f;
    float sumk = 0.0f;

    for (int i = threadIdx.x; i < hdim; i += blockDim.x) {
        sumq += qh[i] * qh[i];
        sumk += kh[i] * kh[i];
    }
    sq[threadIdx.x] = sumq;
    sk[threadIdx.x] = sumk;
    __syncthreads();

    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            sq[threadIdx.x] += sq[threadIdx.x + st];
            sk[threadIdx.x] += sk[threadIdx.x + st];
        }
        __syncthreads();
    }

    float rmsq = rsqrtf(sq[0] / static_cast<float>(hdim) + eps);
    float rmsk = rsqrtf(sk[0] / static_cast<float>(hdim) + eps);

    for (int i = threadIdx.x; i < hdim; i += blockDim.x) {
        qh[i] = qh[i] * rmsq * qw[i];
        kh[i] = kh[i] * rmsk * kw[i];
    }
}

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float x) {
    return static_cast<scalar_t>(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// Fuses the two operations that previously forced three temporary layouts:
// Q/K RMSNorm, RoPE, and the QKV view preparation. QKV is contiguous in
// [sequence, heads, 3, head_dim] order and remains in its original dtype.
template <typename scalar_t>
__global__ void k_fused_qkv_rope_qk_norm(
    scalar_t* qkv,
    const float* qw,
    const float* kw,
    const float* cos_f,
    const float* sin_f,
    int seq,
    int heads,
    int hdim,
    float eps) {
    const int row = blockIdx.x;
    const int total_rows = seq * heads;
    if (row >= total_rows) return;

    const int s = row / heads;
    scalar_t* q = qkv + static_cast<size_t>(row) * 3 * hdim;
    scalar_t* k = q + hdim;

    __shared__ float sq[BLOCK_NORM];
    __shared__ float sk[BLOCK_NORM];
    float qsum = 0.0f;
    float ksum = 0.0f;
    for (int i = threadIdx.x; i < hdim; i += blockDim.x) {
        const float qv = scalar_to_float(q[i]);
        const float kv = scalar_to_float(k[i]);
        qsum += qv * qv;
        ksum += kv * kv;
    }
    sq[threadIdx.x] = qsum;
    sk[threadIdx.x] = ksum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sq[threadIdx.x] += sq[threadIdx.x + stride];
            sk[threadIdx.x] += sk[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float qrms = rsqrtf(sq[0] / static_cast<float>(hdim) + eps);
    const float krms = rsqrtf(sk[0] / static_cast<float>(hdim) + eps);
    for (int i = threadIdx.x; i < hdim; i += blockDim.x) {
        q[i] = float_to_scalar<scalar_t>(scalar_to_float(q[i]) * qrms * qw[i]);
        k[i] = float_to_scalar<scalar_t>(scalar_to_float(k[i]) * krms * kw[i]);
    }
    __syncthreads();

    const int half_dim = hdim / 2;
    for (int d = threadIdx.x; d < half_dim; d += blockDim.x) {
        const int i = d * 2;
        const int freq = s * hdim + i;
        const float c = cos_f[freq];
        const float sn = sin_f[freq];
        const float q0 = scalar_to_float(q[i]);
        const float q1 = scalar_to_float(q[i + 1]);
        const float k0 = scalar_to_float(k[i]);
        const float k1 = scalar_to_float(k[i + 1]);
        q[i] = float_to_scalar<scalar_t>(q0 * c - q1 * sn);
        q[i + 1] = float_to_scalar<scalar_t>(q1 * c + q0 * sn);
        k[i] = float_to_scalar<scalar_t>(k0 * c - k1 * sn);
        k[i + 1] = float_to_scalar<scalar_t>(k1 * c + k0 * sn);
    }
}

__global__ void k_rope_2d_offset(
    float* x,
    const float* cos_f,
    const float* sin_f,
    int seq_len,
    int seq_offset,
    int heads,
    int hdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * heads * (hdim / 2);
    if (idx >= total) return;

    int s = idx / (heads * (hdim / 2));
    int rem = idx % (heads * (hdim / 2));
    int h = rem / (hdim / 2);
    int d = rem % (hdim / 2);

    int freq_idx = s * hdim + d * 2;
    float c = cos_f[freq_idx];
    float sn = sin_f[freq_idx];

    int base = ((seq_offset + s) * heads + h) * hdim + d * 2;
    float x0 = x[base];
    float x1 = x[base + 1];
    x[base] = x0 * c - x1 * sn;
    x[base + 1] = x1 * c + x0 * sn;
}

torch::Tensor silu_mul_cuda_(torch::Tensor gate, torch::Tensor up) {
    auto gate_c = gate.is_contiguous() ? gate : gate.contiguous();
    auto up_c = up.is_contiguous() ? up : up.contiguous();
    int n = static_cast<int>(gate_c.numel());
    int grid = (n + BLOCK_1D - 1) / BLOCK_1D;
    auto stream = at::cuda::getCurrentCUDAStream(gate.get_device()).stream();
    k_silu_mul<<<grid, BLOCK_1D, 0, stream>>>(
        gate_c.data_ptr<float>(),
        up_c.data_ptr<float>(),
        n);
    if (!gate.is_contiguous()) {
        gate.copy_(gate_c);
    }
    return gate;
}

torch::Tensor adaln_norm_cuda(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, double eps) {
    auto x_c = x.contiguous();
    auto shift_c = shift.contiguous();
    auto scale_c = scale.contiguous();
    auto out = torch::empty_like(x_c);
    int seq = static_cast<int>(x_c.size(0));
    int hid = static_cast<int>(x_c.size(1));

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    k_adaln_norm<<<seq, BLOCK_NORM, 0, stream>>>(
        out.data_ptr<float>(),
        x_c.data_ptr<float>(),
        shift_c.data_ptr<float>(),
        scale_c.data_ptr<float>(),
        seq,
        hid,
        static_cast<float>(eps));
    return out;
}

std::vector<torch::Tensor> qk_rms_norm_cuda_(torch::Tensor q, torch::Tensor k, torch::Tensor qw, torch::Tensor kw, double eps) {
    auto q_c = q.is_contiguous() ? q : q.contiguous();
    auto k_c = k.is_contiguous() ? k : k.contiguous();
    auto qw_c = qw.is_contiguous() ? qw : qw.contiguous();
    auto kw_c = kw.is_contiguous() ? kw : kw.contiguous();
    int seq = static_cast<int>(q_c.size(0));
    int heads = static_cast<int>(q_c.size(1));
    int hdim = static_cast<int>(q_c.size(2));
    int grid = seq * heads;

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    k_qk_rms_norm<<<grid, BLOCK_NORM, 0, stream>>>(
        q_c.data_ptr<float>(),
        k_c.data_ptr<float>(),
        qw_c.data_ptr<float>(),
        kw_c.data_ptr<float>(),
        seq,
        heads,
        hdim,
        static_cast<float>(eps));
    if (!q.is_contiguous()) {
        q.copy_(q_c);
    }
    if (!k.is_contiguous()) {
        k.copy_(k_c);
    }
    return {q, k};
}

void fused_qkv_rope_qk_norm_inplace_cuda_(
    torch::Tensor qkv,
    torch::Tensor qw,
    torch::Tensor kw,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t seq_offset,
    int64_t seq_len) {
    TORCH_CHECK(seq_offset == 0, "fused QKV op currently requires seq_offset == 0");
    auto qkv_c = qkv.is_contiguous() ? qkv : qkv.contiguous();
    auto qw_c = qw.is_contiguous() ? qw : qw.contiguous();
    auto kw_c = kw.is_contiguous() ? kw : kw.contiguous();
    auto cos_c = cos.is_contiguous() ? cos : cos.contiguous();
    auto sin_c = sin.is_contiguous() ? sin : sin.contiguous();
    const int seq = static_cast<int>(seq_len);
    const int heads = static_cast<int>(qkv_c.size(1));
    const int hdim = static_cast<int>(qkv_c.size(3));
    const int grid = seq * heads;
    auto stream = at::cuda::getCurrentCUDAStream(qkv_c.get_device()).stream();

    if (qkv_c.scalar_type() == torch::kFloat32) {
        k_fused_qkv_rope_qk_norm<float><<<grid, BLOCK_NORM, 0, stream>>>(
            qkv_c.data_ptr<float>(), qw_c.data_ptr<float>(), kw_c.data_ptr<float>(),
            cos_c.data_ptr<float>(), sin_c.data_ptr<float>(), seq, heads, hdim, 1e-6f);
    } else {
        k_fused_qkv_rope_qk_norm<__nv_bfloat16><<<grid, BLOCK_NORM, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(qkv_c.data_ptr<at::BFloat16>()),
            qw_c.data_ptr<float>(), kw_c.data_ptr<float>(), cos_c.data_ptr<float>(),
            sin_c.data_ptr<float>(), seq, heads, hdim, 1e-6f);
    }
}

torch::Tensor rope_2d_offset_cuda_(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t seq_offset, int64_t seq_len) {
    auto x_c = x.is_contiguous() ? x : x.contiguous();
    auto cos_c = cos.is_contiguous() ? cos : cos.contiguous();
    auto sin_c = sin.is_contiguous() ? sin : sin.contiguous();

    int seq = static_cast<int>(seq_len);
    int heads = static_cast<int>(x_c.size(1));
    int hdim = static_cast<int>(x_c.size(2));
    int total = seq * heads * (hdim / 2);
    int grid = (total + BLOCK_1D - 1) / BLOCK_1D;

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    k_rope_2d_offset<<<grid, BLOCK_1D, 0, stream>>>(
        x_c.data_ptr<float>(),
        cos_c.data_ptr<float>(),
        sin_c.data_ptr<float>(),
        seq,
        static_cast<int>(seq_offset),
        heads,
        hdim);
    if (!x.is_contiguous()) {
        x.copy_(x_c);
    }
    return x;
}

__global__ void k_packed_attention(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int seq,
    int heads,
    int hdim,
    float scale) {
    int idx = blockIdx.x;
    int s = idx / heads;
    int h = idx % heads;
    if (s >= seq) return;

    const float* qh = q + (s * heads + h) * hdim;
    const float* kh = k + (s * heads + h) * hdim;
    const float* vh = v + (s * heads + h) * hdim;
    float* oh = out + (s * heads + h) * hdim;

    __shared__ float partials[ATTN_THREADS];
    __shared__ float score;
    __shared__ float weight;
    __shared__ float rescale;
    __shared__ float prev_rowmax;
    __shared__ float rowmax;
    __shared__ float rowsum;

    if (threadIdx.x == 0) {
        rowmax = -std::numeric_limits<float>::infinity();
        rowsum = 0.0f;
    }
    __syncthreads();

    float acc = 0.0f;
    const bool active = threadIdx.x < hdim;
    const int d = threadIdx.x;
    float vreg = active ? vh[d] : 0.0f;

    const float scaled = scale * LOG2E;
    for (int key = 0; key < seq; ++key) {
        float partial = 0.0f;
        for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
            partial += qh[d] * kh[key * hdim + d];
        }
        partials[threadIdx.x] = partial;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partials[threadIdx.x] += partials[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            prev_rowmax = rowmax;
            score = partials[0] * scaled;
            if (score > rowmax) {
                rescale = exp2f(rowmax - score);
                weight = 1.0f;
                rowsum = rowsum * rescale + 1.0f;
                rowmax = score;
            } else {
                rescale = 1.0f;
                weight = exp2f(score - rowmax);
                rowsum += weight;
            }
        }
        __syncthreads();

        if (active) {
            if (key == 0) {
                acc = vreg;
            } else if (score > prev_rowmax) {
                acc = acc * rescale + vreg;
            } else {
                acc += weight * vreg;
            }
        }
        __syncthreads();
    }

    if (active) {
        oh[d] = acc / rowsum;
    }
}

torch::Tensor packed_attention_cuda_(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale) {
    auto q_c = q.is_contiguous() ? q : q.contiguous();
    auto k_c = k.is_contiguous() ? k : k.contiguous();
    auto v_c = v.is_contiguous() ? v : v.contiguous();
    TORCH_CHECK(q_c.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k_c.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v_c.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(q_c.dim() == 3, "q must be [seq, heads, head_dim]");
    TORCH_CHECK(k_c.sizes() == q_c.sizes() && v_c.sizes() == q_c.sizes(), "q/k/v shapes must match");

    auto out = torch::zeros_like(q_c);
    int seq = static_cast<int>(q_c.size(0));
    int heads = static_cast<int>(q_c.size(1));
    int hdim = static_cast<int>(q_c.size(2));
    int grid = seq * heads;

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    k_packed_attention<<<grid, ATTN_THREADS, 0, stream>>>(
        q_c.data_ptr<float>(),
        k_c.data_ptr<float>(),
        v_c.data_ptr<float>(),
        out.data_ptr<float>(),
        seq,
        heads,
        hdim,
        static_cast<float>(scale));
    return out;
}

torch::Tensor fused_qkv_attention_cuda_(
    torch::Tensor qkv,
    torch::Tensor qw,
    torch::Tensor kw,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t seq_offset,
    int64_t seq_len,
    double scale) {
    auto qkv_c = qkv.is_contiguous() ? qkv : qkv.contiguous();
    TORCH_CHECK(qkv_c.scalar_type() == torch::kFloat32, "qkv must be float32");
    TORCH_CHECK(qkv_c.dim() == 4 && qkv_c.size(2) == 3, "qkv must be [seq, heads, 3, head_dim]");
    auto q = qkv_c.select(2, 0);
    auto k = qkv_c.select(2, 1);
    auto v = qkv_c.select(2, 2);
    qk_rms_norm_cuda_(q, k, qw, kw, 1e-6);
    rope_2d_offset_cuda_(q, cos, sin, seq_offset, seq_len);
    rope_2d_offset_cuda_(k, cos, sin, seq_offset, seq_len);
    return packed_attention_cuda_(q, k, v, scale);
}
