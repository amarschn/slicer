#!/usr/bin/python
"""
Author: Drew Marschner
Created: 8/8/2019
Copyright: (c) Impossible Objects, 2019
"""

from setuptools import setup, find_packages
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("bfg.slice_mesh",sources=["bfg/slice_mesh"+ext])]

if USE_CYTHON:
    extensions = cythonize(extensions)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bfg",
    version="0.0.1",
    description='CBAM Build File Generator (BFG)',
    long_description=long_description,
    url="https://github.com/amarschn/bfg",
    packages=['bfg'],
    classifiers=["Programming Language :: Python :: 3"],
    install_requires=['wheel', 'numpy', 'numpy-stl', 'networkx'],
    ext_modules=extensions,
)