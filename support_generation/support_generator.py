"""
Author: Drew Marschner
Date: 04/04/2019

Support generation for CBAM process

Step by step for support generation:


"""

import numpy as np
from scipy.spatial import ConvexHull
import pyclipper
import matplotlib.pyplot as plt
import stl
import os
import shapely

from OCC.gp import gp_Pnt, gp_Vec
from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Display.SimpleGui import init_display
from ShapeExchange import write_stl_file

import slicer

import ipdb


DEFAULT_SETTINGS = {
    "resolution": 0.5,
    "inner_offset": 6,
    "outer_offset": 10,
    "separator_spacing": 50
}


def grid_points(point_list, spacing=50):
    """
    Returns a set of points that lie on the hull outline
    spaced via a rectilinear grid
    :param" point_list array of xy values
    """

    """
    1. Get bounding box
    2. Fit grid within bounding box - shrink spacing slightly if necessary
    3. Create multilinestring of grid
    4. Create intersection with polygon and multi line string
    5. Flatten intersection data structure into points
    6. Return points
    """
    intersects = []
    poly = shapely.geometry.Polygon(point_list)
    (x_min, y_min, x_max, y_max) = poly.bounds()
    num_vert_lines = int((x_max - x_min)/spacing)
    num_hor_lines = int((y_max - y_min)/spacing)

    return intersects


def generate_convex_support(mesh, resolution, inner_offset=5, outer_offset=10, plot=True):
    """
    Get all slice layers at some resolution
    Create 2D convex hull from set of all slice layers stacked on top of each other
    Create outset from that 2D convex hull
    Extrude outset and boolean region for part
    Export part
    """
    height = float(mesh.z.max())
    sliced_layers = np.array(slicer.layers(mesh, resolution))
    x = np.array([])
    y = np.array([])
    for layer in sliced_layers:
        layer = np.array(layer)
        x = np.append(x, layer[:, :, 0].flatten())
        y = np.append(y, layer[:, :, 1].flatten())
    xy_points = np.array([x, y]).T
    # np.savetxt('h_pts.csv', xy_points)
    # ipdb.set_trace()
    hull = ConvexHull(xy_points)
    # hull_points = tuple(map(tuple, hull.points))
    hull_points = tuple(map(tuple, xy_points[hull.vertices]))
    # ipdb.set_trace()
    # Create offsets
    offset1 = offset_points(hull_points, inner_offset)
    offset1.append(offset1[0])
    offset2 = offset_points(hull_points, outer_offset)
    offset2.append(offset2[0])
    # ipdb.set_trace()
    if plot:
        plt.title('Convex Hull')
        plt.plot(*zip(*hull_points), marker='o', color='b')
        # pp = xy_points[hull.vertices]
        # plt.plot(*zip(*pp), marker='o',color='r',linestyle='None')
        plt.plot(*zip(*offset1), marker='o',color='r')
        plt.plot(*zip(*offset2), marker='o', color='g')
        plt.show()

    inner_edges = []
    outer_edges = []

    inner_wire = BRepBuilderAPI_MakeWire()
    outer_wire = BRepBuilderAPI_MakeWire()

    for i, p in enumerate(offset1[:-1]):
        next_p = offset1[i+1]
        p1 = gp_Pnt(p[0], p[1], 0.)
        p2 = gp_Pnt(next_p[0], next_p[1], 0.)
        # edges.append(BRepBuilderAPI_MakeEdge(p1, p2))
        try:
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            BRepBuilderAPI_MakeWire.Add(inner_wire, edge)
            # inner_edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        except:
            Exception("Edge between {} and {} could not be constructed".format(p, next_p))

    for i, p in enumerate(offset2[:-1]):
        next_p = offset2[i+1]
        p1 = gp_Pnt(p[0], p[1], 0.)
        p2 = gp_Pnt(next_p[0], next_p[1], 0.)
        # edges.append(BRepBuilderAPI_MakeEdge(p1, p2))
        try:
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            BRepBuilderAPI_MakeWire.Add(outer_wire, edge)
            # outer_edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        except:
            Exception("Edge between {} and {} could not be constructed".format(p, next_p))    

    # if inner_edges and outer_edges:
    v = gp_Vec(gp_Pnt(0,0,0), gp_Pnt(0,0,height))

    # w1 = BRepBuilderAPI_MakeWire(*inner_edges)
    f1 = BRepBuilderAPI_MakeFace(inner_wire.Wire())
    p1 = BRepPrimAPI_MakePrism(f1.Face(), v).Shape()

    # w2 = BRepBuilderAPI_MakeWire(*outer_edges)
    f2 = BRepBuilderAPI_MakeFace(outer_wire.Wire())
    p2 = BRepPrimAPI_MakePrism(f2.Face(), v).Shape()

    support_layer = BRepAlgoAPI_Cut(p2, p1).Shape()

    return support_layer


def generate_concave_support(mesh, resolution, k):
    """
    """
    pass


def generate_support(mesh, resolution, output_directory='.', export_stl=True):
    """
    Create support structure and export it if necessary
    """
    support = generate_convex_support(mesh, resolution)
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(support, update=False)
    start_display()
    if export_stl:
        stl_file = os.path.join(output_directory, "support.stl")
        write_stl_file(support, stl_file)
    return support


def offset_points(points, offset, miter_type="SQUARE", miter_limit=5):
    """Accepts a list of points that comprise a closed polygon
    and returns a polygon that is outset from the initial
    polygon by the specified amount
    """

    # Scale by magic number of 1000 to be safe
    # TODO: fix magic number
    scale = 1000

    if miter_type == "SQUARE":
        miter = pyclipper.JT_SQUARE
    elif miter_type == "ROUND":
        miter = pyclipper.JT_ROUND
    elif miter_type == "MITER":
        miter = pyclipper.JT_MITER

    pts = tuple(points)
    pts = pyclipper.scale_to_clipper(pts, scale)
    pco = pyclipper.PyclipperOffset()
    pco.MiterLimit = miter_limit
    pco.AddPath(pts, miter, pyclipper.ET_CLOSEDPOLYGON)

    offset_points = pyclipper.scale_from_clipper(pco.Execute(offset*scale), scale)

    return offset_points[0]

    


def main():
    f = '../test_stl/simple_H.stl'
    # f = '../OpenGL-STL-slicer/nist.stl'
    # f = '../OpenGL-STL-slicer/prism.stl'
    mesh = stl.Mesh.from_file(f)
    generate_support(mesh,0.5)


if __name__ == '__main__':
    main()
