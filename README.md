Build File Generator (BFG)
==========================
Overview
--------
`bfg` is a mesh-slicing, rasterizing program that produces a build file, which is a zipped folder containing files needed to print a build using an Impossible Objects CBAM printer.
The slicing process is primarily based on [`CuraEngine`](https://github.com/Ultimaker/CuraEngine), an open-source slicing library. For more detail on some of the procedures
used in slicing it can be helpful to refer to `CuraEngine` documentation.

Build Settings
--------------
A description of the settings used in a build are below:
```
stl_units :: [MM, IN] :: the units of the mesh vertices given. "MM" represents millimeters, "IN" represents inches
layer_height :: [>0.0] :: the height of single layer. Also called resolution or Z height.
decimal_precision:: the precision at which points are assumed to be the same. E.g. at decimal_precision=3, 1.001 = 1.000
dpi:: [X dpi, Y dpi]::dots per inch, which is axis dependent based on the printhead setup
page_size_in::[X size, Y size]:: The size of the page in inches.
mesh_translation_in::[X translation, Y translation]:: the amount to move the mesh before slicing in X and Y. To be used for mesh placement and centering of the build.
save_zip::[True,False]::whether to write images and other files to a zip or not. 1 indicates yes, 0 indicates no.
zip_output_directory:: The directory in which the zip output will be placed
zip_filename:: the name of the zip file to be output (dependent on the "zip_files" flag).
slice_file_base_name::the base name of a sliced layer image file.
output_directory:: the name of the output directory to be made
image_output_subdirectory:: the name of the image subdirectory
workers:: the number of workers to use in turning slices into images.
edge_to_hole_distance: the distance between the sides of the build and the center points of the nearest alignment holes
page_number_locations::[[]]:: list of lists containing X,Y locations to place page numbers
punch_holes:: legacy hole locations needed for manual printers (Seth).
```

The default settings are listed below:
```json
"stl_units": "MM",
"layer_height": 0.05,
"decimal_precision": 3,
"dpi": [625, 600],
"page_size_in": [11.3, 11.7],
"mesh_translation_in": [-0.35, -0.15, 0.],
"save_zip": True,
"zip_output_directory": '.',
"zip_filename": "output",
"slice_file_base_name": 'layer',
"output_directory": './output/',
"image_output_subdirectory": 'layers/',
"workers": 5,
"edge_to_hole_distance": 0.3,
"page_number_locations": [[0.02, 0.02], [11., 6.]],
"punch_holes": [[7.5, 7.5]]
```

Examples
--------
Using the default settings:
```python
from bfg.build_generator import  BFG

mesh_file = 'example.stl'
B = BFG(mesh_file)
B.create_build()
```

Using non-default settings by loading in new JSON:
```python
from bfg.build_generator import  BFG
import json

with open('example_settings.json', 'r') as f:
    settings = json.load(f)

mesh_file = 'example.stl'
B = BFG(mesh_file, settings)
B.create_build()
```

Slicing Process
---------------
#### Mesh processing:
Once a mesh has been loaded into memory, it must be "optimized" before moving to the slicing step. "Optimizing" here
primarily refers to looking at each mesh face and determining neighbor mesh information, which is useful for speeding
up the actual mesh slicing step.

##### Notes
- Currently the slicer does not perform any mesh repair, and any non-manifold, watertight meshes will result in anomalous output.

#### Mesh slicing:
Once a mesh has been optimized, it is time to slice the mesh. See [here](https://github.com/Ultimaker/CuraEngine/blob/master/docs/slicing.md) for a good description.
The output of the slicing process is a list of layers, where each layer is a list of polygons that must be rasterized.

#### Rasterization:
Once polygons have been determined by the slicing step, they must be rasterized into bitmap images. Rasterization is primarily handled by the `Pillow` Python library

Benchmarks
----------
Below is a table showing the performance of `bfg` in generating builds for various meshes:

| Build     |  Triangles   | Height (mm) | Layer Count (50um) | File size (KB) | Build Time (s) |
|-----------|--------------|-------------|--------------------|----------------|----------------|
| NIST Part | 7392         | 17          | 340                | 362            | 45             |
| Q01 Build | 76288        | 3.2         | 65                 | 3726           | 24             |
| FD Links  | 1560012      | 20          | 400                | 76163          | 435            |

Installation
------------
To install, download the `whl` file in the `dist` folder, and build it using the following command:

`pip install bfg-0.0.1-cp36-cp36m-win_amd64.whl`, where the `whl` file name is replaced with the file name that has been downloaded.

Cython Compilation
-----------
Cython is used in this project for use in speeding up the slicing algorithms.

To compile Cython code:

`
python setup.py build_ext --inplace
`

On MacOSX, type in:

`
export MACOSX_DEPLOYMENT_TARGET=10.9
`

Development
-----------

`
pip install -e .
`

Packaging
---------

`
python setup.py sdist bdist_wheel
`

Copyright Impossible Objects, 2019