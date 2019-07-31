from slicer.image_writer import rasterize
from multiprocessing import Pool
from slicer.optimized_mesh import OptimizedMesh
import slice_mesh_slow
import os


DEFAULT_SETTINGS = {
    "layer_height": 0.2,
    "page_height": 4800,
    "page_width": 7200,
    "output_directory": './output/'
}


class Slicer(object):
    """
    A Slicer object will slice a build
    """
    def __init__(self, mesh_filename, settings=DEFAULT_SETTINGS):
        self.mesh_filename = mesh_filename
        self.settings = settings
        self.optimized_mesh = None
        self.slices = []
        self.mesh_is_processed = False

        if not os.path.exists(self.settings["output_directory"]):
            os.mkdir(self.settings["output_directory"])

    def process_mesh(self):
        self.optimized_mesh = OptimizedMesh(self.mesh_filename)
        self.optimized_mesh.complete()
        self.slices = slice_mesh_slow.slice_mesh(self.optimized_mesh, self.settings["layer_height"])
        self.mesh_is_processed = True

    def create_images(self, workers=5):
        if not self.mesh_is_processed:
            self.process_mesh()
        pool = Pool(workers)
        pool.map(write_layer, self.slices)

    def _linear_create_images(self):
        if not self.mesh_is_processed:
            self.process_mesh()
        for layer in self.slices:
            write_layer(layer)

    def _layer_image(self, layer):
        filename = os.path.join(self.settings["output_directory"], "layer_{}.bmp".format(layer.layer_number))
        layer.make_polygons()
        rasterize(layer.polygons,
                  filename,
                  layer.layer_number,
                  self.settings["page_height"],
                  self.settings["page_width"])

    def layer_image(self, layer):
        if not self.mesh_is_processed:
            self.process_mesh()
        if layer.layer_number > len(self.slices) or layer.layer_number < 0:
            return -1
        else:
            self._layer_image(layer)


def write_layer(layer):
    layer.make_polygons()
    rasterize(layer.polygons, layer.filename, layer.layer_number, layer.height, layer.width)

def main():
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
    s._linear_create_images()

if __name__ == '__main__':
    import cProfile
    cProfile.runctx('main()', globals(), locals(), filename=None)
