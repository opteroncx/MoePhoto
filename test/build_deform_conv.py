import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
module_path = os.path.dirname('.')
setup(
  name='build',
  ext_modules=[
    CUDAExtension(
      name='deform_conv',
      sources=[
        os.path.join(module_path, 'src', 'deform_conv_ext.cpp'),
        os.path.join(module_path, 'src', 'deform_conv_cuda.cpp'),
        os.path.join(module_path, 'src', 'deform_conv_cuda_kernel.cu'),
      ],
      extra_compile_args={'cxx': ['-g', '-O3'],
                          'nvcc': ['-O3']}
    )
  ],
  cmdclass={
    'build_ext': BuildExtension
  })
