#include <torch/extension.h>

void sm120_nvfp4_gemm_out(
    torch::Tensor activation_qdata,
    torch::Tensor activation_scales,
    torch::Tensor weight_qdata_b,
    torch::Tensor weight_scales,
    torch::Tensor output_scale,
    torch::Tensor,
    int64_t kernel_variant);
int64_t sm120_nvfp4_workspace_size(int64_t m, int64_t n, int64_t k, int64_t kernel_variant);

void sm120_nvfp4_gemm_meta(
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    int64_t) {}

TORCH_LIBRARY(klein_sm120, m) {
    m.def(
        "nvfp4_gemm_out(Tensor activation_qdata, Tensor activation_scales, "
        "Tensor weight_qdata_b, Tensor weight_scales, Tensor output_scale, "
        "Tensor(a!) output, int kernel_variant) "
        "-> ()");
}

TORCH_LIBRARY_IMPL(klein_sm120, CUDA, m) {
    m.impl("nvfp4_gemm_out", &sm120_nvfp4_gemm_out);
}

TORCH_LIBRARY_IMPL(klein_sm120, Meta, m) {
    m.impl("nvfp4_gemm_out", &sm120_nvfp4_gemm_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("workspace_size", &sm120_nvfp4_workspace_size);
}
