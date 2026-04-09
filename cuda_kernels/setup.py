from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="klein_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="klein_cuda_ext",
            sources=[
                "src/ops.cpp",
                "src/ops_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
