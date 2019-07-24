from setuptools import setup, find_packages
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("slicer.slice_mesh",
                        sources=["slicer/slice_mesh"+ext])]

if USE_CYTHON:
    extensions = cythonize(extensions)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="slicer",
    version="0.0.1",
    description='Mesh slicer',
    long_description=long_description,
    url="https://github.com/amarschn/slicer",
    packages=['slicer'],
    classifiers=["Programming Language :: Python :: 3"],
    install_requires=['wheel', 'numpy', 'numpy-stl', 'networkx'],
    ext_modules=extensions,
)