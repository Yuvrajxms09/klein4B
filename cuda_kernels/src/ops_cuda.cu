#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
constexpr int BLOCK_NORM = 256;
constexpr int BLOCK_1D = 256;
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
    auto gate_c = gate.contiguous();
    auto up_c = up.contiguous();
    int n = static_cast<int>(gate_c.numel());
    int grid = (n + BLOCK_1D - 1) / BLOCK_1D;
    k_silu_mul<<<grid, BLOCK_1D, 0, at::cuda::getDefaultCUDAStream()>>>(
        gate_c.data_ptr<float>(),
        up_c.data_ptr<float>(),
        n);
    gate.copy_(gate_c);
    return gate;
}

torch::Tensor adaln_norm_cuda(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, double eps) {
    auto x_c = x.contiguous();
    auto shift_c = shift.contiguous();
    auto scale_c = scale.contiguous();
    auto out = torch::empty_like(x_c);
    int seq = static_cast<int>(x_c.size(0));
    int hid = static_cast<int>(x_c.size(1));

    k_adaln_norm<<<seq, BLOCK_NORM, 0, at::cuda::getDefaultCUDAStream()>>>(
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
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto qw_c = qw.contiguous();
    auto kw_c = kw.contiguous();
    int seq = static_cast<int>(q_c.size(0));
    int heads = static_cast<int>(q_c.size(1));
    int hdim = static_cast<int>(q_c.size(2));
    int grid = seq * heads;

    k_qk_rms_norm<<<grid, BLOCK_NORM, 0, at::cuda::getDefaultCUDAStream()>>>(
        q_c.data_ptr<float>(),
        k_c.data_ptr<float>(),
        qw_c.data_ptr<float>(),
        kw_c.data_ptr<float>(),
        seq,
        heads,
        hdim,
        static_cast<float>(eps));
    q.copy_(q_c);
    k.copy_(k_c);
    return {q, k};
}

torch::Tensor rope_2d_offset_cuda_(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t seq_offset, int64_t seq_len) {
    auto x_c = x.contiguous();
    auto cos_c = cos.contiguous();
    auto sin_c = sin.contiguous();

    int seq = static_cast<int>(seq_len);
    int heads = static_cast<int>(x_c.size(1));
    int hdim = static_cast<int>(x_c.size(2));
    int total = seq * heads * (hdim / 2);
    int grid = (total + BLOCK_1D - 1) / BLOCK_1D;

    k_rope_2d_offset<<<grid, BLOCK_1D, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_c.data_ptr<float>(),
        cos_c.data_ptr<float>(),
        sin_c.data_ptr<float>(),
        seq,
        static_cast<int>(seq_offset),
        heads,
        hdim);
    x.copy_(x_c);
    return x;
}
