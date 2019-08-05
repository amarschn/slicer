from slicer.image_writer import rasterize
from multiprocessing import Pool
from slicer.optimized_mesh import OptimizedMesh
import slice_mesh
import os

DEFAULT_SETTINGS = {
    "stl_units": "MM",
    "layer_height": 0.05,
    "decimal_precision": 3,
    "dpi": [625, 600],
    "page_size_in": [11.3, 11.7],
    "mesh_translation_in": [-0.35, -0.15, 0.],
    "slice_file_base_name": 'layer',
    "output_directory": './output/',
    "workers": 5,
    "edge_to_hole_distance": 0.3,
    "page_number_locations": [[0.02, 0.02], [11., 6.]]
}


class Slicer(object):
    """
    A Slicer object will slice a build
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

    def process_mesh(self):
        self.optimized_mesh = OptimizedMesh(self.mesh_filename, self.settings)
        self.optimized_mesh.complete()
        self.slices = slice_mesh.slice_mesh(self.optimized_mesh, self.settings)
        self.mesh_is_processed = True

    def create_images(self):
        """

        :return: None
        """
        if not self.mesh_is_processed:
            self.process_mesh()
        pool = Pool(self.settings["workers"])
        pool.map(write_layer, self.slices)

    def create_build_zip(self):
        """
        Creates a build zip file containing a zipped directory of all images, the settings used stored in a JSON
        file, and the punches.json file
        :return:
        """
        self.create_images()



def write_layer(layer):
    """
    This needs to be a separate function from the class for annoying
    Python reasons: https://thelaziestprogrammer.com/python/a-multiprocessing-pool-pickle


    :param layer:
    :return:
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
    # f = '../test_stl/q01.stl'
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
    f = '../test_stl/q05.stl'
    s = Slicer(f)
    s.create_images()

if __name__ == '__main__':
    import cProfile
    cProfile.runctx('main()', globals(), locals(), filename=None)
