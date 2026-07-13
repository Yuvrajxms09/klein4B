#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = void;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 1;
constexpr int AlignmentD = 8;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using FusionOperation = cutlass::epilogue::fusion::ScaledAcc<
    ElementD,
    ElementCompute,
    ElementCompute>;

template <typename MmaTileShape>
struct KernelConfig {
    using ClusterShape = Shape<_1, _1, _1>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutCTag,
        AlignmentC,
        ElementD,
        LayoutDTag,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        LayoutATag,
        AlignmentA,
        ElementB,
        LayoutBTag,
        AlignmentB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
    using BlockScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

    struct Problem {
        int m;
        int n;
        int k;
        StrideA stride_a;
        StrideB stride_b;
        StrideC stride_c;
        StrideD stride_d;
        LayoutSFA layout_sfa;
        LayoutSFB layout_sfb;
    };

    static Problem make_problem(int m, int n, int k) {
        return {
            m,
            n,
            k,
            cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}),
            cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}),
            cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}),
            cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}),
            BlockScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1)),
            BlockScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1)),
        };
    }

    static typename Gemm::Arguments make_arguments(
        const Problem& problem,
        const void* activation_qdata,
        const void* activation_scales,
        const void* weight_qdata,
        const void* weight_scales,
        const float* output_scale,
        void* output) {
        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {problem.m, problem.n, problem.k, 1},
            {
                static_cast<const ElementA::DataType*>(activation_qdata), problem.stride_a,
                static_cast<const ElementB::DataType*>(weight_qdata), problem.stride_b,
                static_cast<const ElementA::ScaleFactorType*>(activation_scales), problem.layout_sfa,
                static_cast<const ElementB::ScaleFactorType*>(weight_scales), problem.layout_sfb,
            },
            {
                {},
                nullptr,
                problem.stride_c,
                static_cast<ElementD*>(output),
                problem.stride_d,
            },
        };
        auto& fusion = arguments.epilogue.thread;
        fusion.alpha = 0.0f;
        fusion.alpha_ptr = output_scale;
        return arguments;
    }
};

using Kernel128x128x256 = KernelConfig<Shape<_128, _128, _256>>;
using Kernel128x128x128 = KernelConfig<Shape<_128, _128, _128>>;
using Kernel128x64x256 = KernelConfig<Shape<_128, _64, _256>>;
using Kernel128x64x128 = KernelConfig<Shape<_128, _64, _128>>;
using Kernel128x32x256 = KernelConfig<Shape<_128, _32, _256>>;
using Kernel128x32x128 = KernelConfig<Shape<_128, _32, _128>>;

template <typename Config>
int64_t workspace_size_for(int m, int n, int k) {
    using Gemm = typename Config::Gemm;
    const auto problem = Config::make_problem(m, n, k);
    auto arguments = Config::make_arguments(problem, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    return static_cast<int64_t>(Gemm::get_workspace_size(arguments));
}

template <typename Config>
void launch_kernel(
    int m,
    int n,
    int k,
    const void* activation_qdata,
    const void* activation_scales,
    int64_t activation_scale_count,
    const void* weight_qdata,
    const void* weight_scales,
    int64_t weight_scale_count,
    const float* output_scale,
    void* output,
    cudaStream_t stream) {
    using Gemm = typename Config::Gemm;
    const auto problem = Config::make_problem(m, n, k);
    const int64_t expected_sfa = static_cast<int64_t>(cute::size(cute::filter_zeros(problem.layout_sfa)));
    const int64_t expected_sfb = static_cast<int64_t>(cute::size(cute::filter_zeros(problem.layout_sfb)));
    TORCH_CHECK(activation_scale_count == expected_sfa,
                "activation scale layout mismatch: expected ", expected_sfa,
                ", got ", activation_scale_count);
    TORCH_CHECK(weight_scale_count == expected_sfb,
                "weight scale layout mismatch: expected ", expected_sfb,
                ", got ", weight_scale_count);
    auto arguments = Config::make_arguments(
        problem,
        activation_qdata,
        activation_scales,
        weight_qdata,
        weight_scales,
        output_scale,
        output);
    const int64_t required_workspace = static_cast<int64_t>(Gemm::get_workspace_size(arguments));
    TORCH_CHECK(required_workspace == 0, "SM120 NVFP4 kernel unexpectedly requires workspace");
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
                "CUTLASS SM120 NVFP4 kernel rejected this shape");
    TORCH_CHECK(gemm.initialize(arguments, nullptr, stream) == cutlass::Status::kSuccess,
                "CUTLASS SM120 NVFP4 initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
                "CUTLASS SM120 NVFP4 launch failed");
}

template <typename Config>
void launch_from_tensors(
    int m,
    int n,
    int k,
    const torch::Tensor& activation_qdata,
    const torch::Tensor& activation_scales,
    const torch::Tensor& weight_qdata_b,
    const torch::Tensor& weight_scales,
    const torch::Tensor& output_scale,
    const torch::Tensor& output,
    cudaStream_t stream) {
    launch_kernel<Config>(
        m,
        n,
        k,
        activation_qdata.data_ptr(),
        activation_scales.data_ptr(),
        activation_scales.numel(),
        weight_qdata_b.data_ptr(),
        weight_scales.data_ptr(),
        weight_scales.numel(),
        output_scale.data_ptr<float>(),
        output.data_ptr(),
        stream);
}

void check_kernel_variant(int64_t kernel_variant) {
    TORCH_CHECK(kernel_variant >= 0 && kernel_variant <= 5,
                "kernel_variant must be in [0, 5]");
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

int64_t sm120_nvfp4_workspace_size(
    int64_t m64, int64_t n64, int64_t k64, int64_t kernel_variant) {
    TORCH_CHECK(m64 > 0 && n64 > 0 && k64 > 0, "invalid GEMM shape");
    TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "GEMM shape exceeds int32");
    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);
    check_kernel_variant(kernel_variant);
    switch (kernel_variant) {
        case 0: return workspace_size_for<Kernel128x128x256>(m, n, k);
        case 1: return workspace_size_for<Kernel128x128x128>(m, n, k);
        case 2: return workspace_size_for<Kernel128x64x256>(m, n, k);
        case 3: return workspace_size_for<Kernel128x64x128>(m, n, k);
        case 4: return workspace_size_for<Kernel128x32x256>(m, n, k);
        case 5: return workspace_size_for<Kernel128x32x128>(m, n, k);
        default: TORCH_CHECK(false, "unreachable SM120 NVFP4 kernel variant");
    }
    return 0;
}

void sm120_nvfp4_gemm_out(
    torch::Tensor activation_qdata,
    torch::Tensor activation_scales,
    torch::Tensor weight_qdata_b,
    torch::Tensor weight_scales,
    torch::Tensor output_scale,
    torch::Tensor output,
    int64_t kernel_variant) {
    check_cuda_tensor(activation_qdata, "activation_qdata");
    check_cuda_tensor(activation_scales, "activation_scales");
    check_cuda_tensor(weight_qdata_b, "weight_qdata_b");
    check_cuda_tensor(weight_scales, "weight_scales");
    check_cuda_tensor(output_scale, "output_scale");
    check_cuda_tensor(output, "output");
    TORCH_CHECK(activation_qdata.scalar_type() == torch::kUInt8, "activation_qdata must be uint8");
    TORCH_CHECK(weight_qdata_b.scalar_type() == torch::kUInt8, "weight_qdata_b must be uint8");
    TORCH_CHECK(activation_scales.element_size() == 1, "activation_scales must contain FP8 bytes");
    TORCH_CHECK(weight_scales.element_size() == 1, "weight_scales must contain FP8 bytes");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be bfloat16");
    TORCH_CHECK(output_scale.scalar_type() == torch::kFloat32 && output_scale.numel() == 1,
                "output_scale must be one float32 value");
    check_kernel_variant(kernel_variant);
    const auto device = output.device();
    TORCH_CHECK(
        activation_qdata.device() == device && activation_scales.device() == device &&
            weight_qdata_b.device() == device && weight_scales.device() == device &&
            output_scale.device() == device,
        "all SM120 NVFP4 tensors must use the output CUDA device");
    TORCH_CHECK(activation_qdata.dim() == 2 && weight_qdata_b.dim() == 2 && output.dim() == 2,
                "GEMM tensors must be rank 2");

    const int m = static_cast<int>(activation_qdata.size(0));
    const int k = static_cast<int>(activation_qdata.size(1) * 2);
    const int n = static_cast<int>(weight_qdata_b.size(0));
    TORCH_CHECK(weight_qdata_b.size(1) * 2 == k, "weight_qdata_b K mismatch");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    TORCH_CHECK(k % 64 == 0 && n % 2 == 0, "unsupported NVFP4 GEMM alignment");
    c10::cuda::CUDAGuard guard(output.device());
    const auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
    switch (kernel_variant) {
        case 0:
            launch_from_tensors<Kernel128x128x256>(m, n, k, activation_qdata, activation_scales,
                weight_qdata_b, weight_scales, output_scale, output, stream);
            break;
        case 1:
            launch_from_tensors<Kernel128x128x128>(m, n, k, activation_qdata, activation_scales,
                weight_qdata_b, weight_scales, output_scale, output, stream);
            break;
        case 2:
            launch_from_tensors<Kernel128x64x256>(m, n, k, activation_qdata, activation_scales,
                weight_qdata_b, weight_scales, output_scale, output, stream);
            break;
        case 3:
            launch_from_tensors<Kernel128x64x128>(m, n, k, activation_qdata, activation_scales,
                weight_qdata_b, weight_scales, output_scale, output, stream);
            break;
        case 4:
            launch_from_tensors<Kernel128x32x256>(m, n, k, activation_qdata, activation_scales,
                weight_qdata_b, weight_scales, output_scale, output, stream);
            break;
        case 5:
            launch_from_tensors<Kernel128x32x128>(m, n, k, activation_qdata, activation_scales,
                weight_qdata_b, weight_scales, output_scale, output, stream);
            break;
        default: TORCH_CHECK(false, "unreachable SM120 NVFP4 kernel variant");
    }
}
