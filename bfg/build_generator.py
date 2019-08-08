#!/usr/bin/python
"""
Author: Drew Marschner
Created: 8/8/2019
Copyright: (c) Impossible Objects, 2019
"""

from bfg.image_writer import rasterize
from multiprocessing import Pool
from bfg.optimized_mesh import OptimizedMesh
from bfg import slice_mesh
import os
import shutil
import datetime
import json


# TODO: make a documentation generator for documenting all settings
"""
Settings documentation (to be included in general purpose json
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
"""

DEFAULT_SETTINGS = {
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
}

MM_TO_IN = 25.4


class BFG(object):
    """
    The Build File Generator (BFG) object is the main class through which a user will interface with this library.
    """
    def __init__(self, mesh_filename, settings=DEFAULT_SETTINGS):
        self.mesh_filename = mesh_filename
        self.settings = settings
        # Determine the width and length of the image in pixels. This is a pre-computation step
        # to avoid calculating the same values many times at the rasterization step
        self.settings["x_px"] = int(settings["page_size_in"][0] * settings["dpi"][0])
        self.settings["y_px"] = int(settings["page_size_in"][1] * settings["dpi"][1])

        self.optimized_mesh = None
        self.slices = []
        self.mesh_is_processed = False

        if not os.path.exists(self.settings["output_directory"]):
            os.mkdir(self.settings["output_directory"])
        if not os.path.exists(os.path.join(self.settings["output_directory"], self.settings["image_output_subdirectory"])):
            os.mkdir(os.path.join(self.settings["output_directory"], self.settings["image_output_subdirectory"]))

    def _process_mesh(self):
        """
        Internal function used for creating an optimized mesh, which is needed for slicing and rasterizing.
        :return: None
        """
        self.optimized_mesh = OptimizedMesh(self.mesh_filename, self.settings)
        self.optimized_mesh.complete()
        self.slices = slice_mesh.slice_mesh(self.optimized_mesh, self.settings)
        self.mesh_is_processed = True

    def _create_images(self):
        """
        Internal function used for slicing and rasterizing.
        :return: None
        """
        if not self.mesh_is_processed:
            self._process_mesh()
        pool = Pool(self.settings["workers"])
        pool.map(write_layer, self.slices)

    def create_build(self):
        """
        Creates a build zip file containing a zipped directory of all images, the settings used stored in a JSON
        file, and the punches.json file
        :return: None
        """
        # Create images
        self._create_images()

        # Save the settings into a json file
        info_json_fname = os.path.join(self.settings["output_directory"], "info.json")
        with open(info_json_fname, 'w') as f:
            self.settings["input_filename"] = self.mesh_filename
            self.settings["timestamp"] = str(datetime.datetime.now())
            json.dump(self.settings, f)

        # Save the punches.json file, which is only used in legacy manual printers
        if self.settings.get("punch_holes"):
            punch_file = os.path.join(self.settings["output_directory"], "punches.json")
            with open(punch_file, 'w') as f:
                yrows = []
                for x in self.settings["punch_holes"]:
                    yrows.append([i / MM_TO_IN for i in x])
                json.dump(yrows, f)

        # Create zip file
        if self.settings["save_zip"]:
            archive_name = os.path.join(self.settings["zip_output_directory"], self.settings["zip_filename"])
            shutil.make_archive(archive_name, 'zip', self.settings["output_directory"])
            # Delete the raw files and directory
            shutil.rmtree(self.settings["output_directory"])


def write_layer(layer):
    """
    Creates an image of a layer

    This needs to be a separate function from the class for annoying
    Python reasons: https://thelaziestprogrammer.com/python/a-multiprocessing-pool-pickle

    :param layer: a Layer class instance.
    :return: None
    """
    layer.make_polygons()
    rasterize(layer.polygons,
              layer.layer_number,
              layer.settings)

def main():
    # f = '../test_stl/calibration_shapes.stl'
    # f = '../test_stl/1inx1inx200um_brick_origin.stl'
    # f = '../test_stl/1inx1inx200um_brick.stl'
    # f = '../test_stl/logo.stl'
    f = '../test_stl/q01.stl'
    # f = '../test_stl/cylinder.stl'
    # f = '../test_stl/prism.stl'
    # f = '../test_stl/step_square.stl'
    # f = '../test_stl/nist.stl'
    # f = '../test_stl/hollow_prism.stl'
    # f = '../test_stl/10_side_hollow_prism.stl'
    # f = '../test_stl/concentric_1.stl'
    # f = '../test_stl/links.stl'
    # f = '../test_stl/square_cylinder.stl'
    # f = '../test_stl/prism_hole.stl'
    # f = '../test_stl/holey_prism.stl'
    # f = '../test_stl/q05.stl'
    build_generator = BFG(f)
    build_generator.create_build()

if __name__ == '__main__':
    import cProfile
    cProfile.runctx('main()', globals(), locals(), filename=None)