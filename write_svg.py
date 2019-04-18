import numpy as np
# import ipdb


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

    points = np.array(polygon)[0,:,0]
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
            if idx < len(points):
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
    svg = """<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\" 
    xmlns:svg=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" 
    viewport-fill=\"black\">\n
    <!-- Generated using Drew's Slicer -->\n
    """.format(width, height)

    for polygon in polygons:

        if polygon_orientation(polygon) == -1:
            color = "black"
        else:
            color = "white"

        poly_str = "\n<polygon style=\"fill: {}\" points=\"".format(color)

        for segment in polygon:
            x_start, y_start = segment[0]
            x_end, y_end = segment[1]
            poly_str += "{},{} ".format(x_start,y_start,x_end,y_end)
        poly_str += "\"></polygon>"

        svg += poly_str
    svg += "</svg>"
    return svg


if __name__ == '__main__':
    import pickle
    with open('nist1', 'rb') as fp:
        polygons = pickle.load(fp)

    svg = layer_svg(polygons, 1)
    with open('layer.svg', 'w') as f:
        f.write(svg)
