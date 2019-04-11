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

from OCC.gp import gp_Pnt, gp_Vec
from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Display.SimpleGui import init_display

import slicer

import ipdb


def generate_support(mesh, resolution, export_part=True):
    """
    Slice the model by some Z-resolution
    For each sliced layer:
        Create a convex hull of the layer points
        Create 2 offsets of the convex hull based on
        pre-determined offset length
        Extrude the created outline by the Z-resolution height
    Combine all extruded offset layers into one support structure
    """
    sliced_layers = slicer.layers(mesh, resolution)
    support = None
    for layer in sliced_layers:
        support_layer = generate_support_layer(np.array(layer), extrude = resolution)
        if support == None:
            support = support_layer
        else:
            support = BRepAlgoAPI_Fuse(support, support_layer).Shape()

    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(support, update=False)
    start_display()
    if export_part:
        pass
    return support


def generate_support_layer(layer,
                           inner_offset=2,
                           outer_offset=6,
                           plot=True,
                           extrude=0.5):
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

    layer_z = layer[0][0][2]
    extrude_start = layer_z
    extrude_end = layer_z + extrude

    xy_points = np.array([layer[:, :, 0].flatten(), layer[:, :, 1].flatten()]).T
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
        plt.title('Layer Height {}'.format(layer_z))
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
        p1 = gp_Pnt(p[0], p[1], extrude_start)
        p2 = gp_Pnt(next_p[0], next_p[1], extrude_start)
        # edges.append(BRepBuilderAPI_MakeEdge(p1, p2))
        try:
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            BRepBuilderAPI_MakeWire.Add(inner_wire, edge)
            # inner_edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        except:
            Exception("Edge between {} and {} could not be constructed".format(p, next_p))

    for i, p in enumerate(offset2[:-1]):
        next_p = offset2[i+1]
        p1 = gp_Pnt(p[0], p[1], extrude_start)
        p2 = gp_Pnt(next_p[0], next_p[1], extrude_start)
        # edges.append(BRepBuilderAPI_MakeEdge(p1, p2))
        try:
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            BRepBuilderAPI_MakeWire.Add(outer_wire, edge)
            # outer_edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        except:
            Exception("Edge between {} and {} could not be constructed".format(p, next_p))    

    # if inner_edges and outer_edges:
    # ipdb.set_trace()
    v = gp_Vec(gp_Pnt(0,0,extrude_start), gp_Pnt(0,0,extrude_end))

    # w1 = BRepBuilderAPI_MakeWire(*inner_edges)
    f1 = BRepBuilderAPI_MakeFace(inner_wire.Wire())
    p1 = BRepPrimAPI_MakePrism(f1.Face(), v).Shape()

    # w2 = BRepBuilderAPI_MakeWire(*outer_edges)
    f2 = BRepBuilderAPI_MakeFace(outer_wire.Wire())
    p2 = BRepPrimAPI_MakePrism(f2.Face(), v).Shape()

    support_layer = BRepAlgoAPI_Cut(p2, p1).Shape()

    return support_layer
    # else:
    #     return 0


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
    f = '../OpenGL-STL-slicer/nist.stl'
    # f = '../OpenGL-STL-slicer/prism.stl'
    mesh = stl.Mesh.from_file(f)
    generate_support(mesh, 1.)


if __name__ == '__main__':
    main()
