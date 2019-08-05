Slicer
======
Overview
--------
slicer is a mesh-slicing program that outputs bitmap images for use in a 3D-printer.

Currently the slicer does not perform any mesh repair, and any non-manifold, watertight meshes will result in anomalous output.

Process:

Scaling -> pixel off-by-one error

Installation
------------
To install, 

Example
-------


Compilation
-----------
To compile Cython code:

python setup.py build_ext --inplace

On MacOSX, type in:

export MACOSX_DEPLOYMENT_TARGET=10.9

Development
-----------
pip install -e .

Packaging
---------
python setup.py sdist bdist_wheel