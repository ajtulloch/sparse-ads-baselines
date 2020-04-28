from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name='table_batched_embeddings',
      ext_modules=[
          CUDAExtension(
              name='table_batched_embeddings',
              sources=[
                  'table_batched_embeddings.cpp',
                  'table_batched_embeddings_cuda.cu'
              ])
      ],
      cmdclass={'build_ext': BuildExtension})
