import numpy as np
from collections import deque
import cairosvg
import cv2

def polygon_orientation(polygon):
    """
    TODO: this should be implemented in the initial polygon construction,
          thus avoiding going over the list of points twice
    Calculated by determining the minimum y of the polygon, and then choose
    whichever minimum (if there is a tie) has the maximum x, this will
    guarantee that the point lies on the convex hull of the polygon. Once
    this minimum point is found calculate the sign of the orientation matrix
    between the three segments that the point is a part of

    https://en.wikipedia.org/wiki/Curve_orientation

    Returns a boolean
    -1 indicates counter clockwise
    1 indicates clockwise
    """
    min_y = polygon[0][0][1]
    min_y_x = polygon[0][0][0]

    points = np.array(polygon)[:,0,:]
    left_neighbor = points[-1]
    right_neighbor = points[1]

    for idx, pt in enumerate(points):
        x = pt[0]
        y = pt[1]

        if (y < min_y) or (y == min_y and x > min_y_x):
            min_y = y
            min_y_x = x
            if idx > 0:
                left_neighbor = points[idx - 1]
            else:
                left_neighbor = points[len(points)]
            if idx < len(points) - 1:
                right_neighbor = points[idx + 1]
            else:
                right_neighbor = points[0]
    xa = right_neighbor[0]
    ya = right_neighbor[1]
    xb = min_y_x
    yb = min_y
    xc = left_neighbor[0]
    yc = left_neighbor[1]
    det = (xb - xa)*(yc - ya) - (xc - xa)*(yb - ya)
    direction = np.sign(det)
    if direction == 0:
        raise Exception("Unknown Polygon Direction")
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


def arrange_polygons(polygons):
    """
    TODO - make a polygon class, so the winding order doesn't need to be determined multiple times
    Returns holes and not-holes
    :param polygons:
    :return:
    """
    printed_polygons = []
    holes = []
    for polygon in polygons:
        orientation = polygon_orientation(polygon)
        if orientation == -1:
            printed_polygons.append(polygon)
        else:
            holes.append(polygon)
    return printed_polygons, holes


def cv2_rasterize(polygons, height=4800, width=7200, layer=-1, output="cv2_raster.bmp"):
    """

    :param polygons:
    :param height:
    :param width:
    :param output:
    :return:
    """
    # for _ in range(64):

    printed_polygons, holes = arrange_polygons(polygons)

    printed_img = np.zeros([height, width])
    printed_img.fill(255)

    for polygon in polygons:
        color = 0
        points = np.array(polygon)[:, 0, :] * 24.606
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(printed_img, np.int32([points]), color)

    for hole in holes:
        points = np.array(hole)[:, 0, :] * 24.606
        points = points.reshape((-1, 1, 2))
        color = 255
        cv2.fillPoly(printed_img, np.int32([points]), color)

    cv2.putText(img=printed_img,
                text=str(layer),
                org=(250, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3,
                color=(0, 0, 0),
                thickness=15,
                lineType=1,
                bottomLeftOrigin=False)

    cv2.imwrite(output, printed_img)

def cv2_rasterize2(P):
    """

    :param polygons:
    :param height:
    :param width:
    :param output:
    :return:
    """
    # for _ in range(64):

    printed_polygons, holes = arrange_polygons(P.polygons)

    printed_img = np.zeros([P.height, P.width])
    printed_img.fill(255)

    for polygon in printed_polygons:
        color = 0
        points = np.array(polygon)[:, 0, :] * 24.606
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(printed_img, np.int32([points]), color)

    for hole in holes:
        points = np.array(hole)[:, 0, :] * 24.606
        points = points.reshape((-1, 1, 2))
        color = 255
        cv2.fillPoly(printed_img, np.int32([points]), color)

    cv2.putText(img=printed_img,
                text=P.layer,
                org=(250, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3,
                color=(0, 0, 0),
                thickness=15,
                lineType=1,
                bottomLeftOrigin=False)

    cv2.imwrite(P.output_file, printed_img)


def f(name):
    print('hello', name)


class Polygon(object):
    def __init__(self, polygons, layer):
        self.polygons = polygons
        self.layer = str(layer)
        self.height = 4800
        self.width = 7200
        self.output_file = "{}.png".format(self.layer)


if __name__ == '__main__':
    import pickle
    with open('q01', 'rb') as fp:
        polygons = pickle.load(fp)
    Polygons = []
    for i in range(64):
        Polygons.append(Polygon(polygons, i))
    # svg = layer_svg(polygons, 1)
    # with open('layer.svg', 'w') as f:
    #     f.write(svg)
    # output = './layer.png'
    # svg = svg.encode('utf-8')
    # cairosvg.svg2png(svg, write_to=output, dpi=600)
    # import cProfile
    # cProfile.run('cv2_rasterize(polygons, 4800, 7200)')

    ############################################################
    # 12s for bmp, 16s for png
    # for i in range(64):
    #     output = "{}.bmp".format(i)
    #     cv2_rasterize(polygons, 4800, 7200, i, output)
    ############################################################

    #############################################################
    # 30s for bmp, 34s for png
    # from multiprocessing import Pool, Process
    # for i in range(64):
    #     output = "{}.png".format(i)
    #     p = Process(target=cv2_rasterize, args=(polygons, 4800, 7200, i, output))
    #     p.start()
    #     p.join()
    ###############################################################

    ###############################################################
    # 6s for bmp, 6.5s for png
    from multiprocessing import Pool

    p = Pool(5)
    p.map(cv2_rasterize2, Polygons)
    ################################################################