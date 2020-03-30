from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name='sparse_embedding_cuda',
      ext_modules=[
          CUDAExtension(
              name='sparse_embedding_cuda',
              sources=[
                  'sparse_embedding_cuda.cpp',
                  'sparse_embedding_cuda_kernel.cu'
              ],
              include_dirs=[
                  "/private/home/tulloch/src/nccl/2.7.0/src/nccl/build/include",
                  "/public/apps/openmpi/4.0.2/gcc.7.4.0/include"
              ],
              library_dirs=["/public/apps/openmpi/4.0.2/gcc.7.4.0/lib", "/private/home/tulloch/src/nccl/2.7.0/src/nccl/build/lib"],
              libraries=["mpi", "nccl"])
      ],
      cmdclass={'build_ext': BuildExtension})
