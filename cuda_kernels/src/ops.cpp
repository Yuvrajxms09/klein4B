#include <torch/extension.h>
#include <vector>

torch::Tensor silu_mul_cuda_(torch::Tensor gate, torch::Tensor up);
torch::Tensor adaln_norm_cuda(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, double eps);
std::vector<torch::Tensor> qk_rms_norm_cuda_(torch::Tensor q, torch::Tensor k, torch::Tensor qw, torch::Tensor kw, double eps);
torch::Tensor rope_2d_offset_cuda_(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t seq_offset, int64_t seq_len);

torch::Tensor silu_mul_(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA tensor");
    TORCH_CHECK(up.is_cuda(), "up must be CUDA tensor");
    TORCH_CHECK(gate.scalar_type() == torch::kFloat32, "gate must be float32");
    TORCH_CHECK(up.scalar_type() == torch::kFloat32, "up must be float32");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up shapes must match");
    return silu_mul_cuda_(gate, up);
}

torch::Tensor adaln_norm(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(shift.is_cuda(), "shift must be CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(shift.scalar_type() == torch::kFloat32, "shift must be float32");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be [seq, hidden]");
    TORCH_CHECK(shift.dim() == 1 && scale.dim() == 1, "shift/scale must be [hidden]");
    TORCH_CHECK(shift.size(0) == x.size(1) && scale.size(0) == x.size(1), "shift/scale size mismatch");
    return adaln_norm_cuda(x, shift, scale, eps);
}

torch::Tensor qk_rms_norm_(torch::Tensor q, torch::Tensor k, torch::Tensor qw, torch::Tensor kw, double eps) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && qw.is_cuda() && kw.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(qw.scalar_type() == torch::kFloat32, "qw must be float32");
    TORCH_CHECK(kw.scalar_type() == torch::kFloat32, "kw must be float32");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "q and k must be [seq, heads, head_dim]");
    TORCH_CHECK(q.sizes() == k.sizes(), "q and k shapes must match");
    TORCH_CHECK(qw.dim() == 1 && kw.dim() == 1, "qw and kw must be [head_dim]");
    TORCH_CHECK(qw.size(0) == q.size(2) && kw.size(0) == q.size(2), "qw/kw size mismatch");
    qk_rms_norm_cuda_(q, k, qw, kw, eps);
    return q;
}

torch::Tensor rope_2d_offset_(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t seq_offset, int64_t seq_len) {
    TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat32, "cos must be float32");
    TORCH_CHECK(sin.scalar_type() == torch::kFloat32, "sin must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be [seq, heads, head_dim]");
    TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "cos/sin must be [rope_seq, head_dim]");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin shapes must match");
    TORCH_CHECK(cos.size(1) == x.size(2), "cos/sin head_dim mismatch");
    TORCH_CHECK(seq_offset >= 0, "seq_offset must be >= 0");
    TORCH_CHECK(seq_len > 0, "seq_len must be > 0");
    TORCH_CHECK(seq_offset + seq_len <= x.size(0), "seq_offset + seq_len exceeds x sequence length");
    TORCH_CHECK(seq_len <= cos.size(0), "seq_len exceeds cos/sin sequence length");
    return rope_2d_offset_cuda_(x, cos, sin, seq_offset, seq_len);
}

TORCH_LIBRARY(klein_cuda, m) {
    m.def("silu_mul_(Tensor(a!) gate, Tensor up) -> Tensor(a!)");
    m.def("adaln_norm(Tensor x, Tensor shift, Tensor scale, float eps=1e-6) -> Tensor");
    m.def("qk_rms_norm_(Tensor(a!) q, Tensor(b!) k, Tensor qw, Tensor kw, float eps=1e-6) -> Tensor(a!)");
    m.def("rope_2d_offset_(Tensor(a!) x, Tensor cos, Tensor sin, int seq_offset, int seq_len) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(klein_cuda, CUDA, m) {
    m.impl("silu_mul_", &silu_mul_);
    m.impl("adaln_norm", &adaln_norm);
    m.impl("qk_rms_norm_", &qk_rms_norm_);
    m.impl("rope_2d_offset_", &rope_2d_offset_);
}
