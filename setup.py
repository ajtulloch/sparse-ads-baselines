from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name="table_batched_embeddings",
    ext_modules=[
        CUDAExtension(
            name="table_batched_embeddings",
            include_dirs=["/private/home/tulloch/src/"],
            sources=[
                "table_batched_embeddings.cpp",
                "table_batched_embeddings_cuda.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3"
                ],
                "nvcc": [
                    "-lineinfo",
                    "-O3",
                    # '--resource-usage',
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

