from image_writer import cv2_rasterize, pillow_rasterize
import os


def test_pil(all_polygons):
    for layer_number, polygons in enumerate(all_polygons):
        output = os.path.join('./output', "layer_{}.bmp".format(layer_number))
        pillow_rasterize(polygons, output, layer_number, 4800, 7600)

def test_cairo(all_polygons):
    pass


def test_cv2(all_polygons):
    for layer_number, polygons in enumerate(all_polygons):
        output = os.path.join('./output', "layer_{}.bmp".format(layer_number))
        cv2_rasterize(polygons, output, layer_number, 4800, 7600)


def test_pyglet(all_polygons):
    pass


if __name__ == "__main__":
    import pickle

    with open('all_polygons.pickle', 'rb') as fp:
        all_polygons = pickle.load(fp)

    import cProfile
    cProfile.runctx('test_pil(all_polygons)', globals(), locals(), filename=None)