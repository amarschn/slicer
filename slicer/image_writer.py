"""Module for exporting image files from the polygons created by slicer_object.py.

Author: Drew Marschner
"""

import numpy as np
from collections import deque
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
    -1 indicates counter clockwise
    1 indicates clockwise
    """
    if len(polygon) < 3:
        return -1

    min_y = polygon[0][1]
    min_y_x = polygon[0][0]

    left_neighbor = polygon[-1]
    right_neighbor = polygon[1]

    for idx, pt in enumerate(polygon):
        x = pt[0]
        y = pt[1]

        if (y < min_y) or (y == min_y and x > min_y_x):
            min_y = y
            min_y_x = x
            if idx > 0:
                left_neighbor = polygon[idx - 1]
            else:
                left_neighbor = polygon[len(polygon) - 1]
            if idx < len(polygon) - 1:
                right_neighbor = polygon[idx + 1]
            else:
                right_neighbor = polygon[0]

    xa = right_neighbor[0]
    ya = right_neighbor[1]
    xb = min_y_x
    yb = min_y
    xc = left_neighbor[0]
    yc = left_neighbor[1]
    det = (xb - xa)*(yc - ya) - (xc - xa)*(yb - ya)
    direction = np.sign(det)
    if direction == 0:
        # print("Polygon: {}".format(polygon))
        # print("Minimum point: {}, {}".format(min_y_x, min_y))
        # raise Exception("Unknown Polygon Direction")
        print("Unknown Polygon Direction")
    return direction


def layer_svg(polygons, layer, width=304.8, height=203.2):
    """
    """
    svg = """<svg width=\"{}mm\" height=\"{}mm\" units=\"mm\" xmlns=\"http://www.w3.org/2000/svg\" 
    xmlns:svg=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" 
    viewport-fill=\"black\">\n
    <!-- Generated using Drew's Slicer -->\n
    """.format(width, height)

    svg += "<rect x=\"0\" y=\"0\" style=\"stroke-width:1; stroke:rgb(0,0,0)\" width=\"304.8\" height=\"203.2\" fill=\"white\"></rect>"
    all_poly_str = deque([])

    for polygon in polygons:
        orientation = polygon_orientation(polygon)
        if orientation == -1:
            color = "black"
        else:
            color = "white"

        poly_str = "\n<polygon units=\"mm\" style=\"fill: {}\" points=\"".format(color)

        for segment in polygon:
            x_start, y_start = segment[0]
            x_end, y_end = segment[1]
            poly_str += "{},{} ".format(x_start,y_start,x_end,y_end)
        poly_str += "\"></polygon>"

        if orientation == -1:
            all_poly_str.appendleft(poly_str)
        else:
            all_poly_str.append(poly_str)
    svg += ''.join(all_poly_str)
    text = "<text x=\"{}\" y=\"{}\" fill=\"black\">{}</text>".format(width/2, height, layer)
    svg += text
    svg += "</svg>"
    return svg


def save_png(svg_string, output_file):
    """
    :param svg_string:
    :return:
    """
    svg = svg_string.encode('utf-8')
    cairosvg.svg2png(svg, write_to=output_file)


def polygon_size(polygon):
    """
    Returns bounding box area of a polygon
    """
    np_polygon = np.array(polygon)

    bbox_x = np.max(np_polygon[:,0]) - np.min(np_polygon[:,0])
    bbox_y = np.max(np_polygon[:,1]) - np.min(np_polygon[:,1])

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