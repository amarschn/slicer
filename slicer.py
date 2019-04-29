import stl
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from image_writer import cv2_rasterize
import os
from multiprocessing import Pool


def layer_graph(segments, decimal_place=3):
    """
    This function orders all line segments and returns an array of polygons,
    where a polygon is an array of points [(x1, y1), (x2, y2), ...]. This is
    accomplished by the following steps:

    Make a digraph with edges representing from all segments
    Remove the bridges from digraph
    Return the cycles (aka polygons) of the digraph with all bridges removed.
    
    Useful links:
    - https://stackoverflow.com/questions/48736396/algorithm-to-find-bridges-from-cut-vertices
    - https://visualgo.net/en/dfsbfs

    :param segments:
    :param decimal_place:
    :return:
    """
    D = nx.DiGraph()
    for seg in np.round(segments, decimals=decimal_place):
        p1 = tuple(seg[0])
        p2 = tuple(seg[1])

        if p1 == p2:
            pass
        else:
            D.add_edge(p1, p2)

    H = nx.Graph(D)
    B = nx.bridges(H)

    for b in B:
        D.remove_edge(b[0], b[1])

    C = nx.simple_cycles(D)

    return list(C)


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
    # slice_list = []
    for i, layer in enumerate(layers):
        sl = Slice(layer_number=i)
        slices.append(sl)
        # slice_list.append([])

    for triangle in mesh:
        p0 = triangle[0:3]
        p1 = triangle[3:6]
        p2 = triangle[6:9]

        (z0, z1, z2) = p0[2], p1[2], p2[2]

        for layer_num, z in enumerate(layers):
            segment = []
            if z < min(z0,z1,z2):
                continue
            elif z > max(z0,z1,z2):
                continue
            elif z0 < z and z1 >= z and z2 >= z:
                # What condition is this?
                segment = calculate_segment(p0, p2, p1, z)

            elif z0 > z and z1 < z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p0, p1, p2, z)

            elif z0 >= z and z1 < z and z2 >= z:
                # What condition is this?
                segment = calculate_segment(p1, p0, p2, z)

            elif z0 < z and z1 > z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p1, p2, p0, z)

            elif z0 >= z and z1 >= z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p2, p1, p0, z)

            elif z0 < z and z1 < z and z2 > z:
                # What condition is this?
                segment = calculate_segment(p2, p0, p1, z)

            else:
                # Not all cases create a segment
                continue

            if segment:
                slices[layer_num].add_segment(segment)
                # slice_list[layer_num].append(segment)
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


def segment_test():
    p1 = (0, 0)
    p2 = (10, 0)
    p3 = (10, 10)
    p4 = (0, 10)
    p5 = (15, 25)
    p6 = (25, 35)

    s1 = [p1, p2]
    s2 = [p2, p3]
    s3 = [p3, p5]
    s4 = [p3, p4]
    s5 = [p4, p1]
    s6 = [p5, p3]
    s7 = [p5, p6]
    s8 = [p6, p5]
    segments = [s1, s2, s3, s4, s5, s6, s7, s8]
    return segments

def plot_polygon_points(polygons):
    """
    Expects an array of [(x1,y1), (x2,y2), ... ] arrays and will plot those array as a digraph of points to points
    :param polygon_points:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for polygon_pts in polygons:
        for i, point in enumerate(polygon_pts):
            x0 = point[0]
            y0 = point[1]
            if i == len(polygon_pts) - 1:
                next_idx = 0
            else:
                next_idx = i+1

            x1 = polygon_pts[next_idx][0]
            y1 = polygon_pts[next_idx][1]

            ax.plot(x0, y0, '.', lineStyle='None')
            ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.05, length_includes_head=True)
    plt.show()


def write_layer(S):
    unordered_segments = S.segments
    polygons = layer_graph(unordered_segments)
    print(S.layer_number)
    
    cv2_rasterize(polygons=polygons,
                  output_file=S.filename,
                  layer=S.layer_number,
                  height=S.height,
                  width=S.width)

    # plot_polygon_points(polygons)


class Slice(object):
    def __init__(self, segments=None, layer_number=-1, height=4800, width=7200):
        self.filename = os.path.join('./output', "layer_{}.png".format(layer_number))
        if segments is None:
            self.segments = []
        else:
            self.segments = segments
        self.layer_number = layer_number
        self.height = 4800
        self.width = 7200

    def add_segment(self, segment):
        self.segments.append(segment)

def main():
    # f = './test_stl/logo.stl'
    f = './test_stl/q01.stl'
    # f = './test_stl/cylinder.stl'
    # f = './test_stl/prism.stl'
    # f = './test_stl/nist.stl'
    # f = './test_stl/hollow_prism.stl'
    # f = './test_stl/10_side_hollow_prism.stl'
    # f = './test_stl/concentric_1.stl'
    mesh = stl.Mesh.from_file(f)
    resolution = 1.0
    Slices = get_unordered_slices(mesh, resolution)
    for Slice in Slices:
        write_layer(Slice)
    # pool = Pool(5)
    # pool.map(write_layer, Slices)



    # for i, s in enumerate(slices):
    #     print(i)
    #     polygons = layer_graph(s)
    #     vector_layers[i] = polygons
        # plot_polygon_points(polygons)

    # with open('vector_layers.json', 'w') as f:
    #     json.dump(vector_layers, f)



if __name__ == '__main__':
    # import cProfile
    # cProfile.runctx('main()', globals(), locals(), filename=None)
    main()
    # plt.arrow(0,0,2,2, head_width=0.05, length_includes_head=True)
    # plt.show()
