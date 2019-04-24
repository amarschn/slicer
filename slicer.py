import stl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import pprint
from collections import deque
import ipdb


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
        ax.plot(x, y, '.', lineStyle='None')

        x0 = s[0][0]
        x1 = s[1][0]
        y0 = s[0][1]
        y1 = s[1][1]
        ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.05, length_includes_head=True)
    plt.show()


def plot_polygons(polygons):
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for poly in polygons:
        for seg in poly:
            # ipdb.set_trace()
            x = [seg[0][0], seg[1][0]]
            y = [seg[0][1], seg[1][1]]
            ax.plot(x, y, '.', lineStyle='None')

            x0 = seg[0][0]
            x1 = seg[1][0]
            y0 = seg[0][1]
            y1 = seg[1][1]
            ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.05, length_includes_head=True)
    plt.show()


def make_polygons(segments, decimal_place=3):
    pt_dict = {}


    for seg in np.round(segments, decimals=decimal_place):
        p1 = tuple(seg[0])
        p2 = tuple(seg[1])

        if p1 == p2:
            print("Segment is a point: {} : {}".format(p1, p2))
            continue

        if p1 in pt_dict:
            if pt_dict[p1][1] != None:
                print("P1 point already in dict: {}".format(p1))
                print(p1, p2)
                if pt_dict[p1][0] ==  pt_dict[p1][1]:
                    print("Loop detected at point: {}".format(p1))
                # continue
                continue

            
            
            pt_dict[p1][1] = p2
        else:
            pt_dict[p1] = [None, p2]

        if p2 in pt_dict:
            if pt_dict[p2][0] != None:
                print("P2 point already in dict: {}".format(p2))
                print(p1, p2)
                if pt_dict[p2][0] == pt_dict[p2][1]:
                    print("Loop detected at point: {}".format(p2))
                continue

            pt_dict[p2][0] = p1
        else:
            pt_dict[p2] = [p1, None]


    polygons = []

    while len(pt_dict) > 0:
        polygon = []
        first_pt, [_, next_pt] = pt_dict.popitem()
        polygon.append([first_pt, next_pt])

        while first_pt != next_pt:
            current_pt = next_pt
            try:
                _, next_pt = pt_dict.pop(current_pt)
            except:
                print("Failed to pop {}".format(current_pt))
                plot_polygons(polygons)
            polygon.append([current_pt, next_pt])
        polygons.append(polygon)

    return polygons




def naive_make_polygons(s, tol = .005):
    """
    Takes in all the segments of a layer, and orders them such that every
    segment is connected at both ends to another segment, and no segment
    end is connected to more than 1 other segment end.

    This will can create multiple polygons

    TODO: is this even necessary for a bitmap-based output? Answer - Pretty much yes.
    TODO: implement Bentley Ottmann algorithm to take this from O(n2) to O(n log n)
    """
    unaltered_segs = set(map(tuple, s))
    segments = unaltered_segs.copy()
    # print("Segment count: ", len(segments))
    polygons = []
    D = deque([segments.pop()])
    while segments:
        # Create beginning and end of chain
        start = D[0][0]
        end = D[-1][1]
        # print(D, len(D))
        for seg in unaltered_segs:
            if seg not in segments:
                continue

            seg_start = seg[0]
            seg_end = seg[1]
            # if the first point of the first segment is equal to the second
            # point of the second segment, then add the second segment to the
            # top of the deque
            if np.isclose(start, seg_end, atol = tol).all():
                segments.remove(seg)
                D.appendleft(seg)
                start = seg_start
            elif np.isclose(end, seg_start, atol = tol).all():
                segments.remove(seg)
                D.append(seg)
                end = seg_end
        if len(D) > 1 and np.isclose(start, end, atol = tol).all():
            print("Ordered segment count: ", len(D))
            polygons.append(D)
            if segments:
                D = deque([segments.pop()])
    return polygons


def get_unordered_slices(mesh, resolution):
    """
    For each triangle:
        Get points
        For each layer:
            Check if layer intersects triangle
            
    :param mesh:
    :param resolution:
    :return:
    """
    height = mesh.z.max() - mesh.z.min()
    layers = np.array([z for z in range(int(height/resolution) + 1)])*resolution

    slices = []
    for layer in enumerate(layers):
        slices.append([])

    for triangle in mesh:
        p0 = triangle[0:3]
        p1 = triangle[3:6]
        p2 = triangle[6:9]

        (z0, z1, z2) = p0[2], p1[2], p2[2]
        # (y0, y1, y2) = round(p0[1]), round(p1[1]), round(p2[1])
        # if y0 == y1 == y2 == 0:
        #     print triangle

        for layer_num, z in enumerate(layers):
            segment = []
            if z < min(z0,z1,z2):
                continue
            elif z > max(z0,z1,z2):
                continue
            elif z0 < z and z1 >= z and z2 >= z:
                # What condition is this?
                segment = calculate_segment(p0, p2, p1, z)
                # slices[z].append(segment)
            elif z0 > z and z1 < z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p0, p1, p2, z)
                # slices[z].append(segment)
            elif z0 >= z and z1 < z and z2 >= z:
                # What condition is this?
                segment = calculate_segment(p1, p0, p2, z)
                # slices[z].append(segment)
            elif z0 < z and z1 > z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p1, p2, p0, z)
                # slices[z].append(segment)
            elif z0 >= z and z1 >= z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p2, p1, p0, z)
                # slices[z].append(segment)
            elif z0 < z and z1 < z and z2 > z:
                # What condition is this?
                segment = calculate_segment(p2, p0, p1, z)
                # slices[z].append(segment)
            else:
                # Not all cases create a segment
                continue
            if segment:
                # if segment[1][0] == 6:
                    # ipdb.set_trace()
                slices[layer_num].append(segment)
    return slices


def calculate_segment(p0, p1, p2, z):
    """
    """
    # ipdb.set_trace()
    x_start = interpolate(z, p0[2], p1[2], p0[0], p1[0])
    x_end = interpolate(z, p0[2], p2[2], p0[0], p2[0])

    y_start = interpolate(z, p0[2], p1[2], p0[1], p1[1])
    y_end = interpolate(z, p0[2], p2[2], p0[1], p2[1])
    
    return [(x_start, y_start), (x_end, y_end)]


def interpolate(y, y0, y1, x0, x1):
    """
    """
    dx = x1 - x0
    dy = y1 - y0

    # If the slope is negative
    if dy < 0:
        # the proportion of the curve we are interpolating when the slope is negative is flipped
        p = (y - max(y0, y1)) / dy
        x = dx * p + x0
    else:
        p = (y - min(y0, y1)) / dy
        x = dx * p + x0
    return x


def main():
    f = './test_stl/logo.stl'
    # f = './test_stl/q01.stl'
    # f = './test_stl/cylinder.stl'
    # f = './test_stl/prism.stl'
    # f = './test_stl/nist.stl'
    # f = './test_stl/hollow_prism.stl'
    # f = './test_stl/10_side_hollow_prism.stl'
    mesh = stl.Mesh.from_file(f)
    resolution = 1.0
    slices = get_unordered_slices(mesh, resolution)
    # for i, s in enumerate(slices):
    #     print(i)
    #     polygons = make_polygons(s)
    # ipdb.set_trace()
    # print segments
    # print segments.shape
    # ipdb.set_trace()
    # print segments[10]
    # polygons = make_polygons(slices[2])
    # print(polygons)
    plot_individual_segments(slices[2])
    # plot_polygons(polygons)

    # import pickle

    # with open('q01', 'wb') as fp:
    #     pickle.dump(polygons, fp)


if __name__ == '__main__':
    import cProfile
    # cProfile.runctx('main()', globals(), locals(), filename=None)
    main()
    # plt.arrow(0,0,2,2, head_width=0.05, length_includes_head=True)
    # plt.show()
