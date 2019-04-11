import stl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb
import math
import pprint
from collections import deque


def get_intersect(segment, layer):
    """
    Returns tuple intersect point of a line segment and a given layer height
    """
    # If the segment top or bottom intersects the layer, return that point
    if layer > segment[0][2] and layer > segment[1][2]:
        return ()
    elif layer < segment[0][2] and layer < segment[1][2]:
        return ()
    elif segment[0][2] == layer:
        return tuple(segment[0])
    elif segment[1][2] == layer:
        return tuple(segment[1])
    else:
        dx = segment[1][0] - segment[0][0]
        dy = segment[1][1] - segment[0][1]
        dz = segment[1][2] - segment[0][2]
        p = abs((layer - min(segment[0][2], segment[1][2]))/dz)

        # Account for segments that point "down" or "up" in Z
        if dz > 0:
            ix = segment[0][0] + dx*p
            iy = segment[0][1] + dy*p
        else:
            ix = segment[1][0] - dx*p
            iy = segment[1][1] - dy*p

        return (ix, iy, layer)


def plot_all_segments(segments):
    """
    Uses matplotlib to plot all line segments in slice
    Assumes that segments are an np array of shape (N, 3, 2)
    """
    x = segments[:, :, 0].flatten()
    y = segments[:, :, 1].flatten()
    z = segments[:, :, 2].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.plot(x, y, '.-')
    plt.show()


def plot_individual_segments(segments):
    """
    Plots individual segments as separate colors
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for s in segments:
        x = [s[0][0], s[1][0]]
        y = [s[0][1], s[1][1]]
        ax.plot(x, y, '.-')
    plt.show()


def brute_order_segments(segments):
    """
    Takes in all the segments of a layer, and orders them such that every
    segment is connected at both ends to another segment, and no segment
    end is connected to more than 1 other segment end

    TODO: is this even necessary for a bitmap-based output?
    """
    D = deque([segments.pop()])
    # ipdb.set_trace()
    while segments:
        # Create beginning and end of chain
        start = D[0][0]
        end = D[-1][1]
        for seg in segments:
            seg_start = seg[0]
            seg_end = seg[1]
            # if the first point of the first segment is equal to the second
            # point of the second segment, then add the second segment to the
            # top of the deque
            if np.isclose(start, seg_start).all():
                segments.remove(seg)
                seg = [(seg_end, seg_start)]
                D.appendleft(seg)
            elif np.isclose(start, seg_end).all():
                segments.remove(seg)
                D.appendleft(seg)
            elif np.isclose(end, seg_end).all():
                segments.remove(seg)
                seg = [(seg_end, seg_start)]
                D.append(seg)
            elif np.isclose(end, seg_start).all():
                segments.remove(seg)
                D.append(seg)
    return D


def layers(mesh, resolution):
    height = mesh.z.max() - mesh.z.min()
    slices = np.array([z for z in range(int(height/resolution) + 1)])*resolution
    layers = []
    for layer in slices:
        segments = get_layer_segments(mesh, layer)
        if segments:
            layers.append(segments)
    return layers

def get_layer_segments(mesh, layer):
    # For each slice, loop through all triangles and determine if they intersect
    # with that slice
    segments = []
    for triangle in mesh:
        p1 = triangle[0:3]
        p2 = triangle[3:6]
        p3 = triangle[6:]
        z = np.array([triangle[2], triangle[5], triangle[8]])
        # If the layer is too high or too low for the triangle, don't worry
        # about slicing, this triangle is out of the relevant range
        if layer >= max(z) or layer <= min(z):
            continue
        # If the triangle has points both above and below the z, create
        # segments from the intersection of the triangle and the layer

        # Some rules about intersections:
        # There can only be 2 intersections of a triangle for a valid slice, no more and no less
        # Single intersections correspond to a tangent or point meeting of the triangle to a slice plane
        # Infinite intersections correspond to a situation where the triangle or triangle segment is parallel to the slice plane

        # For each segment, find if there is an intersection with the slice plane
        # Calculate the intersection via interpolation
        seg1 = np.array([p1, p2])
        seg2 = np.array([p2, p3])
        seg3 = np.array([p3, p1])
        segs = [seg1, seg2, seg3]
        # print segs
        # print "\n\n"
        intersects = []

        for seg in segs:
            intersect = get_intersect(seg, layer)
            if intersect:
                intersects.append(intersect)
        if len(intersects) == 2:
            segments.append(intersects)
    return segments


def main():
    f = '../OpenGL-STL-slicer/nist.stl'
    # f = '../OpenGL-STL-slicer/prism.stl'
    mesh = stl.Mesh.from_file(f)
    resolution = 10.1
    sliced_layers = layers(mesh, resolution)
    for layer in sliced_layers:
        plot_individual_segments(layer)
    # print len(sliced_layers)
    # print("Unordered segments: ", segments)
    # for s in segments:
    #     print s
    # print("Ordered segments: ", brute_order_segments(segments))
    # plot_individual_segments(np.array(segments))


if __name__ == '__main__':
    import cProfile
    cProfile.runctx('main()', globals(), locals(), filename=None)
    # main()
