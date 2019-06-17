"""Set up script for compiling cython code."""


from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
    Extension("optimized_mesh_old",
              sources=["optimized_mesh_old.pyx"],
              include_dirs=[numpy.get_include()],
              language="c++",
              extra_compile_args=["-stdlib=libc++", "-fpic"])
]

setup(
    ext_modules=cythonize(extensions)
)
