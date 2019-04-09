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

import ipdb
from OCC.gp import gp_Pnt, gp_Vec
from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Display.SimpleGui import init_display


LAYERS = np.array([[(15.0, 20.0, 15.0), (9.0, 12.0, 15.0)],
                  [(9.0, 12.0, 15.0), (0., 0., 15.0)],
                  [(0., 0., 15.0), (6.0, 0., 15.0)],
                  [(6.0, 0., 15.0), (15.0, 0., 15.0)],
                  [(15.0, 0., 15.0), (15.0, 8.0, 15.0)],
                  [(15.0, 8.0, 15.0), (15.0, 20.0, 15.0)]])


def generate_support(mesh):
    """
    Slice the model by some Z-resolution
    For each sliced layer:
        Create a convex hull of the layer points
        Create 2 offsets of the convex hull based on
        pre-determined offset length
        Extrude the created outline by the Z-resolution height
    Combine all extruded offset layers into one support structure
    """
    return 0


def generate_support_layer(layer, inner_offset=2, outer_offset=6, plot=False, extrude=0.5):
    """
    Create a convex hull of the layer points
        Create 2 offsets of the convex hull based on
        pre-determined offset length
        Extrude the created outline by the Z-resolution height
    """
    # Make sure points are in a format of np.array([(x1,y1,z1), (x2,y2,z2)])
    if layer.shape != (layer.shape[0], 2, 3):
        print layer.shape
        raise Exception('Layer is not formatted correctly')
    # ipdb.set_trace()
    xy_points = np.array([layer[:, :, 0].flatten(), layer[:, :, 1].flatten()]).T
    hull = ConvexHull(xy_points)
    hull_points = tuple(map(tuple, hull.points))

    # Create offsets
    offset1 = offset_points(hull_points, inner_offset)
    offset1.append(offset1[0])
    offset2 = offset_points(hull_points, outer_offset)
    offset2.append(offset2[0])

    if plot:
        plt.plot(*zip(*xy_points))
        plt.plot(*zip(*offset1))
        plt.plot(*zip(*offset2))
        plt.show()

    inner_edges = []
    outer_edges = []
    for i, p in enumerate(offset1[:-1]):
        next_p = offset1[i+1]
        p1 = gp_Pnt(p[0], p[1], 0.)
        p2 = gp_Pnt(next_p[0], next_p[1], 0.)
        # edges.append(BRepBuilderAPI_MakeEdge(p1, p2))
        try:
            inner_edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        except:
            Exception("Edge between {} and {} could not be constructed".format(p, next_p))

    for i, p in enumerate(offset2[:-1]):
        next_p = offset2[i+1]
        p1 = gp_Pnt(p[0], p[1], 0.)
        p2 = gp_Pnt(next_p[0], next_p[1], 0.)
        # edges.append(BRepBuilderAPI_MakeEdge(p1, p2))
        try:
            outer_edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        except:
            Exception("Edge between {} and {} could not be constructed".format(p, next_p))    

    if inner_edges and outer_edges:
        v = gp_Vec(gp_Pnt(0,0,0), gp_Pnt(0,0,extrude))

        w1 = BRepBuilderAPI_MakeWire(*inner_edges)
        f1 = BRepBuilderAPI_MakeFace(w1.Wire())
        p1 = BRepPrimAPI_MakePrism(f1.Face(), v).Shape()

        w2 = BRepBuilderAPI_MakeWire(*outer_edges)
        f2 = BRepBuilderAPI_MakeFace(w2.Wire())
        p2 = BRepPrimAPI_MakePrism(f2.Face(), v).Shape()

        support_layer = BRepAlgoAPI_Cut(p2, p1).Shape()

        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(support_layer, update=False)
        start_display()
        return p
    else:
        return 0


def offset_points(points, offset, miter_type="MITER", miter_limit=5):
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
    # generate_support()

    generate_support_layer(LAYERS, 1, 2)


if __name__ == '__main__':
    main()
