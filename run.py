import slicer
import image_writer
import stl


f = './test_stl/cylinder.stl'
mesh = stl.Mesh.from_file(f)
resolution = 0.05
slices = slicer.get_unordered_slices(mesh, resolution)
for num, layer in enumerate(slices):
    print(num)
    if layer:
        fname = './images/layer_{}.png'.format(num)
        polygons = slicer.make_polygons(layer)
        svg = image_writer.layer_svg(polygons, fname)
        image_writer.save_png(svg, fname)

