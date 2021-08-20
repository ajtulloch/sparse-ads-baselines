from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name='table_batched_embeddings',
      ext_modules=[
          CUDAExtension(
              name='table_batched_embeddings',
              include_dirs=['/home/m092926/daisy/Documents/pytorch/aten/src', '/home/m092926/daisy/Documents/cub'],
              sources=[
                  'table_batched_embeddings.cpp',
                  'table_batched_embeddings_cuda.cu'
              ])
      ],
      cmdclass={'build_ext': BuildExtension},
      data_files=[('', ['table_batched_embeddings_ops.py'])]
)
