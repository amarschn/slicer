"""Module for exporting image files from the polygons created by slicer_object.py.

Author: Drew Marschner
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# FONT_BOLD = ImageFont.truetype(os.path.join(os.path.dirname(tkioutils.__file__), tkioutils.FONT_BOLD), 100)


def transform_polygon(polygon, transform):
    """
    Transform a set of polygons according to a 3x3 transform
    :param image:
    :param transform:
    :return:
    """
    polygon = np.array(polygon)

    # Set to homogeneous coordinates
    polygon = np.append(polygon, np.ones([len(polygon),1]), axis=1)

    polygon = np.dot(polygon, transform)

    return polygon[:, 0:2]


def polygon_orientation(polygon):
    """
    TODO: this should be implemented in the initial polygon construction,
          thus avoiding going over the list of points twice
    Calculated by determining the minimum y of the polygon, and then choose
    whichever minimum (if there is a tie) has the maximum x, this will
    guarantee that the point lies on the convex hull of the polygon. Once
    this minimum point is found calculate the sign of the orientation matrix
    between the two segments that the point is a part of

    https://en.wikipedia.org/wiki/Curve_orientation

    Returns a boolean
    -1 indicates counter clockwise, which will be polymer
    1 indicates clockwise, which will be empty
    """
    if len(polygon) < 3:
        return -1

    hull_pt_idx = 0

    for idx, pt in enumerate(polygon):
        x = pt[0]
        y = pt[1]
        if (y < polygon[hull_pt_idx][1]) or \
           (y == polygon[hull_pt_idx][1] and x > polygon[hull_pt_idx][0]):
            hull_pt_idx = idx

    right_neighbor = neighbor_point(polygon, hull_pt_idx, "RIGHT")
    left_neighbor = neighbor_point(polygon, hull_pt_idx, "LEFT")

    xa = right_neighbor[0]
    ya = right_neighbor[1]
    xb = polygon[hull_pt_idx][0]
    yb = polygon[hull_pt_idx][1]
    xc = left_neighbor[0]
    yc = left_neighbor[1]
    det = (xb - xa)*(yc - ya) - (xc - xa)*(yb - ya)
    direction = np.sign(det)
    if direction == 0:
        # TODO: make better error handling for this case
        print("Unknown Polygon Direction: {}".format(polygon))
        return -1
    return direction


def neighbor_point(polygon, pt_idx, direction):
    """
    Determine a non-copy point that is the right or left neighbor of a point in a list
    of tuple points representing a 2D polygon
    :param polygon:
    :param direction: "RIGHT" or "LEFT"
    :return:
    """
    assert direction in ["RIGHT", "LEFT"]
    neighbor_idx = pt_idx

    while polygon[neighbor_idx] == polygon[pt_idx]:
        if direction == "RIGHT":
            neighbor_idx += 1
            if neighbor_idx >= len(polygon):
                neighbor_idx = 0
        elif direction == "LEFT":
            neighbor_idx -= 1
            if neighbor_idx < 0:
                neighbor_idx = len(polygon) - 1

    return polygon[neighbor_idx]



def polygon_size(polygon):
    """
    Returns bounding box area of a polygon
    """
    np_polygon = np.array(polygon)

    bbox_x = np.max(np_polygon[:,0]) - np.min(np_polygon[:, 0])
    bbox_y = np.max(np_polygon[:,1]) - np.min(np_polygon[:, 1])

    return bbox_x * bbox_y


def arrange_polygons(polygons):
    """
    TODO - make a polygon class, so the winding order doesn't need to be determined multiple times
    Returns holes and not-holes
    :param polygons:
    :return:
    """
    arranged_polygons = sorted(polygons, key=polygon_size, reverse=True)
    is_hole = []
    for idx, polygon in enumerate(arranged_polygons):
        orientation = polygon_orientation(polygon)
        if orientation == -1:
            is_hole.append(False)
        elif orientation == 1:
            is_hole.append(True)
        else:
            is_hole.append(True)
    
    return arranged_polygons, is_hole


def rasterize(polygons, output_file, layer, height, width, transform=None):
    """

    :param polygons:
    :param output_file:
    :param layer:
    :param height:
    :param width:
    :param transform:
    :return:
    """
    arranged_polygons, is_hole = arrange_polygons(polygons)
    im = Image.new("1", (width, height), 1)
    draw = ImageDraw.Draw(im)
    for idx, polygon in enumerate(arranged_polygons):
        if len(polygon) < 2:
            continue
        if is_hole[idx]:
            color = 1
        else:
            color = 0

        if transform:
            polygon = transform_polygon(polygon, transform)

        # Scale the polygon to get to the correct DPI
        polygon = np.array(polygon)
        polygon = tuple(map(tuple, polygon * 24.606))
        draw.polygon(polygon, fill=color)

    # Flip and rotate the image...there has to be a better way to do this...
    # Look into transforming the entire STL before slicing
    im = im.transpose(Image.ROTATE_180)
    im = im.transpose(Image.FLIP_LEFT_RIGHT)

    # Add text of layer number
    draw = ImageDraw.Draw(im)
    # draw.text((10, 10), str(layer), fill=0, font=FONT_BOLD)
    im.save(output_file)


class Polygon(object):
    def __init__(self, polygons, layer):
        self.polygons = polygons
        self.layer = str(layer)
        self.height = 4800
        self.width = 7200
        self.output_file = "{}.png".format(self.layer)


if __name__ == '__main__':
    pass