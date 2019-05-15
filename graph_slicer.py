import stl
import numpy as np
import matplotlib.pyplot as plt
from image_writer import cv2_rasterize
import os
import sys
from multiprocessing import Pool
import igraph


class Vertex(object):
    def __init__(self, idx, p):
        self.idx = idx
        self.p = p


class Slice(object):
    """
    A slice of a build, containing all segments within that layer
    """
    def __init__(self,
                 layer_number=-1,
                 height=4800,
                 width=7200,
                 transform=np.eye(3)):
        self.filename = os.path.join('./output', "layer_{}.png".format(layer_number))
        self.layer_number = layer_number
        self.height = 4800
        self.width = 7600
        self.transform = transform
        self.vertex_hash_map = {}
        self.vertices = []
        self.segments = []
        self.edges = []

    def add_segment(self, segment):
        idx0 = self.add_vertex(segment[0])
        idx1 = self.add_vertex(segment[1])
        self.edges.append((idx0, idx1))
        self.segments.append(segment)

    def add_vertex(self, v):
        v_hash = self.point_hash(v)

        # If the vertex hash is already stored, then get the key
        index = self.vertex_hash_map.get(v_hash)

        if index is not None:
            return index
        else:
            index = len(self.vertices)
            self.vertex_hash_map[v_hash] = index
            vertex = Vertex(index, v_hash)
            self.vertices.append(vertex)
            return index

    def point_hash(self, v):
        """
        Returns a hash for the vertex and any other point within
        the meld distance
        """
        v = np.round(v, decimals = 3)
        return tuple(v)


def layer_graph(slice_layer):
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
    graph = igraph.Graph(directed=True)
    graph.add_vertices(len(slice_layer.vertices))
    graph.add_edges(slice_layer.edges)
    x = graph.components()
    return x
    
    


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
    layers = np.array([z for z in range(int(height / resolution) + 1)]) * resolution

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
            if z < min(z0, z1, z2):
                continue
            elif z > max(z0, z1, z2):
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
        ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.5, length_includes_head=True)
    plt.show()


def plot_polygons(polygons):
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
    polygons = layer_graph(S)
    # cv2_rasterize(polygons=polygons,
    #               output_file=S.filename,
    #               layer=S.layer_number,
    #               height=S.height,
    #               width=S.width,
    #               transform=S.transform)

    # plot_polygon_points(polygons)



def main():
    # f = './test_stl/logo.stl'
    # f = './test_stl/q01.stl'
    # f = './test_stl/cylinder.stl'
    f = './test_stl/prism.stl'
    # f = './test_stl/nist.stl'
    # f = './test_stl/hollow_prism.stl'
    # f = './test_stl/10_side_hollow_prism.stl'
    # f = './test_stl/concentric_1.stl'
    # f = './test_stl/links.stl'
    # f = './test_stl/square_cylinder.stl'
    mesh = stl.Mesh.from_file(f)
    resolution = 1.0
    Slices = get_unordered_slices(mesh, resolution)
    for Slice in Slices:
        write_layer(Slice)
    # pool = Pool(5)
    # pool.map(write_layer, Slices)



if __name__ == '__main__':
    # import cProfile
    # cProfile.runctx('main()', globals(), locals(), filename=None)
    main()
    # plt.arrow(0,0,2,2, head_width=0.05, length_includes_head=True)
    # plt.show()