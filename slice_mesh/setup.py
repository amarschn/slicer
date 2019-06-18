"""Set up script for compiling cython code."""


from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
    Extension("slice_mesh",
              sources=["slice_mesh.pyx"],
              include_dirs=[numpy.get_include()],
              language="c++",
              extra_compile_args=["-stdlib=libc++", "-fpic"])
]

setup(
    ext_modules=cythonize(extensions)
)
