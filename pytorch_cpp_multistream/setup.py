from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(name='cpp_extensions',
      ext_modules=[
          CUDAExtension(name='plugins', 
                        sources=['bindings.cpp'],
                        )
          ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
    )
