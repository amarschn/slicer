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
import math

from OCC.gp import gp_Pnt, gp_Vec
from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Display.SimpleGui import init_display
from ShapeExchange import write_stl_file

import slicer
import concave_hull
import image_writer
import ipdb


DEFAULT_SETTINGS = {
    "resolution": 0.5,
    "inner_offset": 6,
    "outer_offset": 10,
    "separator_spacing": 50
}

def simple_blast_holes(mesh):
    """
    Adds simple blast holes to a mesh. Returns the modified mesh
    Several assumptions are made:
    - The mesh is a support wall
    - The mesh does not contain any parts that are not support
    - The wall is roughly convex
    """
    _, center, _ = mesh.get_mass_properties()
    


def create_support(mesh, resolution, offset, wall_thickness):
    """
    Offset the exteriors of all mesh outlines at a given layer resolution
    Boolean all exterior offsets together into one or more polygons
    These polygons form the outer shell(s)
    Offset these polygons again by the desired wall thickness
    Extrude the wall

    Offset the interiors of all mesh outlines at a given layer resolution
    Intersect all interior offsets together into one or more polygons
    These polygons form the inner supports
    Extrude all polygons according to some minimal area metric
    """

    outer_polygons = None
    inner_polygons = None

    z_min = float(mesh.z.min())
    z_max = float(mesh.z.max())

    Slices = slicer.get_unordered_slices(mesh, resolution)
    for sliced_layer in Slices:
        polygons = slicer.layer_graph(sliced_layer.segments, sliced_layer.layer_number)
        
        for polygon in polygons:
            orientation = image_writer.polygon_orientation(polygon)

            if orientation == -1:
                if not outer_polygons:
                    outer_polygons = pyclipper.Pyclipper()
                    outer_polygons.AddPath(tuple(polygon), pyclipper.PT_SUBJECT, True)
                else:
                    try:
                        outer_polygons.AddPath(tuple(polygon), pyclipper.PT_CLIP, True)
                    except pyclipper.ClipperException:
                        continue
            elif orientation == 1:
                if not inner_polygons:
                    inner_polygons = pyclipper.Pyclipper()
                    inner_polygons.AddPath(tuple(polygon), pyclipper.PT_SUBJECT, True)
                else:
                    try:
                        inner_polygons.AddPath(tuple(polygon), pyclipper.PT_CLIP, True)
                    except pyclipper.ClipperException:
                        continue
    if outer_polygons:
        # Union the polygons, then offset
        part_polygons = outer_polygons.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        support_polygons = pyclipper.PyclipperOffset()
        for polygon in part_polygons:
            polygon = tuple(map(tuple, polygon))
            support_polygons.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        outer_support_inner_wall = support_polygons.Execute(offset)
        outer_support_outer_wall = support_polygons.Execute(offset+wall_thickness)
    else:
        outer_support_outer_wall = []
        outer_support_inner_wall = []
    
    if inner_polygons:
        part_polygons = inner_polygons.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        support_polygons = pyclipper.PyclipperOffset()
        for polygon in part_polygons:
            polygon = tuple(map(tuple, polygon))
            support_polygons.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        inner_support_polygons = support_polygons.Execute(-offset)
    else:
        inner_support_polygons = []

    return extrude_supports(outer_support_outer_wall,
                            outer_support_inner_wall,
                            inner_support_polygons,
                            z_min,
                            z_max)



def extrude_supports(outer_support_outer_wall,
                     outer_support_inner_wall,
                     inner_support_polygons,
                     z_min,
                     z_max):
    """
    Accepts list of polygons for both outer and inner support walls [[[x1,y1, ...]], [[x1,y1],...]]
    Returns OCC Shape object
    """
    combined_support = None
    for polygon in outer_support_outer_wall:
        support = extrude_polygon(polygon, z_min, z_max)
        if not combined_support:
            combined_support = support
        else:
            combined_support = BRepAlgoAPI_Fuse(combined_support, support).Shape()

    for polygon in outer_support_inner_wall:
        remove_support = extrude_polygon(polygon, z_min, z_max)
        combined_support = BRepAlgoAPI_Cut(combined_support, remove_support).Shape()

    for polygon in inner_support_polygons:
        support = extrude_polygon(polygon, z_min, z_max)
        if not combined_support:
            combined_support = support
        else:
            combined_support = BRepAlgoAPI_Fuse(combined_support, support).Shape()

    return combined_support


def extrude_polygon(polygon, z_min, z_max):
    """
    Takes in polygon of form [(x1,y1), (x2,y2), ...]
    Returns OCC Shape object
    """
    wire = BRepBuilderAPI_MakeWire()

    for i, p in enumerate(polygon):
        if i < len(polygon) - 1:
            next_p = polygon[i+1]
        else:
            next_p = polygon[0]

        p1 = gp_Pnt(p[0], p[1], 0.)
        p2 = gp_Pnt(next_p[0], next_p[1], 0.)

        try:
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            BRepBuilderAPI_MakeWire.Add(wire, edge)
        except:
            Exception("Edge between {} and {} could not be constructed".format(p, next_p))

    # if inner_edges and outer_edges:
    v = gp_Vec(gp_Pnt(0,0,z_min), gp_Pnt(0,0,z_max))

    # w1 = BRepBuilderAPI_MakeWire(*inner_edges)
    face = BRepBuilderAPI_MakeFace(wire.Wire())
    prism = BRepPrimAPI_MakePrism(face.Face(), v).Shape()

    return prism


def test_clipper_support(export_stl=True, output_directory='.'):
    f = './test_stl/4_parts.stl'
    mesh = stl.Mesh.from_file(f)
    resolution = 20
    support = create_support(mesh, resolution, 6, 6)
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(support, update=False)
    start_display()
    if export_stl:
        stl_file = os.path.join(output_directory, "support.stl")
        write_stl_file(support, stl_file)
        support_mesh = stl.Mesh.from_file(stl_file)
        combined = stl.mesh.Mesh(np.concatenate([mesh.data, support_mesh.data]))
        combined.save('combined.stl')
    return support


def grid_points(points, x_count=50, y_count=50):
    """
    Returns a set of points that represent a "pixelized" view of a set of points
    :param" point_list array of xy values
    """
    x_min, x_max, y_min, y_max = (np.min(points[:,0]), np.max(points[:,0]), np.min(points[:,1]), np.max(points[:,1]))
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pixel_size = x_range / x_count
    y_pixel_size = y_range / y_count
    xlin = np.linspace(x_min + x_pixel_size / 2, x_max - x_pixel_size / 2, x_count)
    ylin = np.linspace(y_min + y_pixel_size / 2, y_max - y_pixel_size / 2, y_count)

    grid = set()

    for pt in points:

        cell_x = abs(int(((pt[0] - x_min) / x_range) * x_count))
        cell_y = abs(int(((pt[1] - y_min) / y_range) * y_count))

        cell_x = min(cell_x, len(xlin) - 1)
        cell_y = min(cell_y, len(ylin) - 1)

        grid_pt = (xlin[cell_x], ylin[cell_y])

        grid.add(grid_pt)

    return np.array(list(grid))


def get_layer_points(mesh, resolution):
    """
    Extract all XY points from a mesh given a resolution
    Requires slicing the model
    """
    slices = slicer.get_unordered_slices(mesh, resolution)
    x = np.array([])
    y = np.array([])
    for layer_slice in slices:
        segs = np.array(layer_slice.segments)
        # ipdb.set_trace()
        if segs.any():
            x = np.append(x, segs[:, 0, 0])
            y = np.append(y, segs[:, 0, 1])
    return np.array([x, y]).T


def generate_support(mesh, resolution, type='convex', inner_offset=6.35, outer_offset=20, plot=True):
    """
    Get all slice layers at some resolution
    Create 2D convex hull from set of all slice layers stacked on top of each other
    Create outset from that 2D convex hull
    Extrude outset and boolean region for part
    Export part
    """
    height = float(mesh.z.max())
    xy_points = get_layer_points(mesh, resolution)
    xy_points = grid_points(xy_points)
    
    if type is 'convex':
        hull = ConvexHull(xy_points)
        # hull_points = tuple(map(tuple, hull.points))
        hull_points = tuple(map(tuple, xy_points[hull.vertices]))
    else:
        max_points = 100000
        hull_points = concave_hull.concave_hull(xy_points, 20)
    # Create offsets
    offset1 = offset_points(hull_points, inner_offset)
    offset1.append(offset1[0])
    offset2 = offset_points(hull_points, outer_offset)
    offset2.append(offset2[0])
    # ipdb.set_trace()
    if plot:
        plt.title('{} Hull'.format(type))
        plt.plot(*zip(*hull_points), marker='o', color='b')
        # pp = xy_points[hull.vertices]
        # plt.plot(*zip(*pp), marker='o',color='r',linestyle='None')
        plt.plot(*zip(*offset1), marker='o', color='r')
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


def offset_points(points, offset, miter_type="ROUND", miter_limit=5):
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
    # f = './test_stl/square_cylinder.stl'
    f = './test_stl/4_Parts.stl'
    # f = './test_stl/4_brackets.STL'
    # f = './test_stl/prism.stl'
    # f = '../OpenGL-STL-slicer/nist.stl'
    # f = '../OpenGL-STL-slicer/prism.stl'
    # f = './test_stl/links.stl'
    # f = './test_stl/simple_H.stl'
    mesh = stl.Mesh.from_file(f)

    # mesh.rotate([0,1,0], math.radians(90))

    resolution = 2.0
    export_stl = True
    output_directory = '.'

    support = generate_support(mesh, resolution, type='concave')
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(support, update=False)
    start_display()
    if export_stl:
        stl_file = os.path.join(output_directory, "support.stl")
        write_stl_file(support, stl_file)
        support_mesh = stl.Mesh.from_file(stl_file)
        combined = stl.mesh.Mesh(np.concatenate([mesh.data, support_mesh.data]))
        combined.save('combined.stl')
    return support


if __name__ == '__main__':
    main()
    # test_clipper_support()