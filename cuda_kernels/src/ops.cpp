#include <torch/extension.h>
#include <vector>

torch::Tensor silu_mul_cuda_(torch::Tensor gate, torch::Tensor up);
torch::Tensor adaln_norm_cuda(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, double eps);
std::vector<torch::Tensor> qk_rms_norm_cuda_(torch::Tensor q, torch::Tensor k, torch::Tensor qw, torch::Tensor kw, double eps);
torch::Tensor rope_2d_offset_cuda_(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t seq_offset, int64_t seq_len);
std::vector<torch::Tensor> fused_qkv_rope_qk_norm_cuda_(
    torch::Tensor qkv,
    torch::Tensor qw,
    torch::Tensor kw,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t seq_offset,
    int64_t seq_len);
torch::Tensor packed_attention_cuda_(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);
torch::Tensor fused_qkv_attention_cuda_(
    torch::Tensor qkv,
    torch::Tensor qw,
    torch::Tensor kw,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t seq_offset,
    int64_t seq_len,
    double scale);
std::vector<torch::Tensor> joint_packed_attention_cuda_(
    torch::Tensor q_hidden,
    torch::Tensor k_hidden,
    torch::Tensor v_hidden,
    torch::Tensor q_context,
    torch::Tensor k_context,
    torch::Tensor v_context,
    double scale);

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

torch::Tensor packed_attention_(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be [seq, heads, head_dim]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shapes must match");
    TORCH_CHECK(q.size(2) <= 128, "packed_attention_ currently supports head_dim <= 128");
    return packed_attention_cuda_(q, k, v, scale);
}

torch::Tensor fused_qkv_attention_(
    torch::Tensor qkv,
    torch::Tensor qw,
    torch::Tensor kw,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t seq_offset,
    int64_t seq_len,
    double scale) {
    TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA tensor");
    TORCH_CHECK(qkv.scalar_type() == torch::kFloat32, "qkv must be float32");
    TORCH_CHECK(qkv.dim() == 4 && qkv.size(2) == 3, "qkv must be [seq, heads, 3, head_dim]");
    TORCH_CHECK(qkv.size(3) <= 128, "fused_qkv_attention_ currently supports head_dim <= 128");
    return fused_qkv_attention_cuda_(qkv, qw, kw, cos, sin, seq_offset, seq_len, scale);
}

std::vector<torch::Tensor> joint_packed_attention_(
    torch::Tensor q_hidden,
    torch::Tensor k_hidden,
    torch::Tensor v_hidden,
    torch::Tensor q_context,
    torch::Tensor k_context,
    torch::Tensor v_context,
    double scale) {
    TORCH_CHECK(q_hidden.is_cuda() && k_hidden.is_cuda() && v_hidden.is_cuda(), "hidden tensors must be CUDA");
    TORCH_CHECK(q_context.is_cuda() && k_context.is_cuda() && v_context.is_cuda(), "context tensors must be CUDA");
    TORCH_CHECK(q_hidden.scalar_type() == torch::kFloat32, "q_hidden must be float32");
    TORCH_CHECK(k_hidden.scalar_type() == torch::kFloat32, "k_hidden must be float32");
    TORCH_CHECK(v_hidden.scalar_type() == torch::kFloat32, "v_hidden must be float32");
    TORCH_CHECK(q_context.scalar_type() == torch::kFloat32, "q_context must be float32");
    TORCH_CHECK(k_context.scalar_type() == torch::kFloat32, "k_context must be float32");
    TORCH_CHECK(v_context.scalar_type() == torch::kFloat32, "v_context must be float32");
    TORCH_CHECK(q_hidden.dim() == 3 && q_context.dim() == 3, "q tensors must be [seq, heads, head_dim]");
    TORCH_CHECK(k_hidden.sizes() == q_hidden.sizes() && v_hidden.sizes() == q_hidden.sizes(), "hidden shapes must match");
    TORCH_CHECK(k_context.sizes() == q_context.sizes() && v_context.sizes() == q_context.sizes(), "context shapes must match");
    TORCH_CHECK(q_hidden.sizes().slice(1) == q_context.sizes().slice(1), "head shapes must match");

    auto k_cat = torch::cat({k_context.contiguous(), k_hidden.contiguous()}, 0);
    auto v_cat = torch::cat({v_context.contiguous(), v_hidden.contiguous()}, 0);
    auto out_context = packed_attention_cuda_(q_context, k_cat, v_cat, scale);
    auto out_hidden = packed_attention_cuda_(q_hidden, k_cat, v_cat, scale);
    return {out_hidden, out_context};
}

std::vector<torch::Tensor> fused_qkv_rope_qk_norm_(
    torch::Tensor qkv,
    torch::Tensor qw,
    torch::Tensor kw,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t seq_offset,
    int64_t seq_len) {
    TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA tensor");
    TORCH_CHECK(qkv.scalar_type() == torch::kFloat32, "qkv must be float32");
    TORCH_CHECK(qkv.dim() == 4, "qkv must be [seq, heads, 3, head_dim]");
    TORCH_CHECK(qkv.size(2) == 3, "qkv third dimension must be 3");
    TORCH_CHECK(qw.is_cuda() && kw.is_cuda() && cos.is_cuda() && sin.is_cuda(), "aux tensors must be CUDA");
    TORCH_CHECK(qw.scalar_type() == torch::kFloat32, "qw must be float32");
    TORCH_CHECK(kw.scalar_type() == torch::kFloat32, "kw must be float32");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat32, "cos must be float32");
    TORCH_CHECK(sin.scalar_type() == torch::kFloat32, "sin must be float32");
    TORCH_CHECK(qw.dim() == 1 && kw.dim() == 1, "qw/kw must be [head_dim]");
    TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "cos/sin must be [rope_seq, head_dim]");
    TORCH_CHECK(seq_offset >= 0, "seq_offset must be >= 0");
    TORCH_CHECK(seq_len > 0, "seq_len must be > 0");

    auto qkv_c = qkv.contiguous();
    int seq = static_cast<int>(seq_len);
    int heads = static_cast<int>(qkv_c.size(1));
    int hdim = static_cast<int>(qkv_c.size(3));
    auto q = qkv_c.select(2, 0).contiguous();
    auto k = qkv_c.select(2, 1).contiguous();
    auto v = qkv_c.select(2, 2).contiguous();

    qk_rms_norm_cuda_(q, k, qw, kw, 1e-6);
    rope_2d_offset_cuda_(q, cos, sin, seq_offset, seq);
    rope_2d_offset_cuda_(k, cos, sin, seq_offset, seq);
    return {q, k, v};
}

TORCH_LIBRARY(klein_cuda, m) {
    m.def("silu_mul_(Tensor(a!) gate, Tensor up) -> Tensor(a!)");
    m.def("adaln_norm(Tensor x, Tensor shift, Tensor scale, float eps=1e-6) -> Tensor");
    m.def("qk_rms_norm_(Tensor(a!) q, Tensor(b!) k, Tensor qw, Tensor kw, float eps=1e-6) -> Tensor(a!)");
    m.def("rope_2d_offset_(Tensor(a!) x, Tensor cos, Tensor sin, int seq_offset, int seq_len) -> Tensor(a!)");
    m.def("fused_qkv_rope_qk_norm_(Tensor qkv, Tensor qw, Tensor kw, Tensor cos, Tensor sin, int seq_offset, int seq_len) -> Tensor[]");
    m.def("packed_attention_(Tensor q, Tensor k, Tensor v, float scale=1.0) -> Tensor");
    m.def("fused_qkv_attention_(Tensor qkv, Tensor qw, Tensor kw, Tensor cos, Tensor sin, int seq_offset, int seq_len, float scale=1.0) -> Tensor");
    m.def("joint_packed_attention_(Tensor q_hidden, Tensor k_hidden, Tensor v_hidden, Tensor q_context, Tensor k_context, Tensor v_context, float scale=1.0) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(klein_cuda, CUDA, m) {
    m.impl("silu_mul_", &silu_mul_);
    m.impl("adaln_norm", &adaln_norm);
    m.impl("qk_rms_norm_", &qk_rms_norm_);
    m.impl("rope_2d_offset_", &rope_2d_offset_);
    m.impl("fused_qkv_rope_qk_norm_", &fused_qkv_rope_qk_norm_);
    m.impl("packed_attention_", &packed_attention_);
    m.impl("fused_qkv_attention_", &fused_qkv_attention_);
    m.impl("joint_packed_attention_", &joint_packed_attention_);
}
