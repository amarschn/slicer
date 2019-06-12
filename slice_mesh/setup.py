from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["slice_mesh.pyx", "optimized_mesh.pyx"]),
)
