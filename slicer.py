import stl
import numpy as np
import matplotlib.pyplot as plt
from image_writer import cv2_rasterize
import os
import sys
from multiprocessing import Pool
from optimized_mesh import OptimizedMesh
import queue
import networkx as nx

CONNECTED_GAP = 0.01


def slice_mesh(optimized_mesh, resolution):
    """
    For each triangle:
        Get points
        For each layer:
            Check if layer intersects triangle

    :param mesh:
    :param resolution:
    :return:
    """
    height = optimized_mesh.mesh.z.max() - optimized_mesh.mesh.z.min()
    layers = np.array([z for z in range(int(height / resolution) + 1)]) * resolution

    slices = []

    for i, layer in enumerate(layers):
        sl = Slice(layer_number=i)
        slices.append(sl)

    for face in optimized_mesh.faces:
        p0 = optimized_mesh.vertices[face.vertex_indices[0]].p
        p1 = optimized_mesh.vertices[face.vertex_indices[1]].p
        p2 = optimized_mesh.vertices[face.vertex_indices[2]].p

        (z0, z1, z2) = p0[2], p1[2], p2[2]

        for layer_num, z in enumerate(layers):
            segment = []
            end_vertex = None
            if z < min(z0, z1, z2):
                continue
            elif z > max(z0, z1, z2):
                continue
            elif z0 < z and z1 >= z and z2 >= z:
                # What condition is this?
                segment = calculate_segment(p0, p2, p1, z)
                end_edge_idx = 0
                if p1[2] == z:
                    end_vertex = optimized_mesh.vertices[face.vertex_indices[1]]

            elif z0 > z and z1 < z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p0, p1, p2, z)
                end_edge_idx = 2

            elif z0 >= z and z1 < z and z2 >= z:
                # What condition is this?
                segment = calculate_segment(p1, p0, p2, z)
                end_edge_idx = 1
                if p2[2] == z:
                    end_vertex = optimized_mesh.vertices[face.vertex_indices[2]]

            elif z0 < z and z1 > z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p1, p2, p0, z)
                end_edge_idx = 0

            elif z0 >= z and z1 >= z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p2, p1, p0, z)
                end_edge_idx = 2
                if p0[2] == z:
                    end_vertex = optimized_mesh.vertices[face.vertex_indices[0]]

            elif z0 < z and z1 < z and z2 > z:
                # What condition is this?
                segment = calculate_segment(p2, p0, p1, z)
                end_edge_idx = 1

            else:
                # Not all cases create a segment
                continue

            if segment:
                sliced_layer = slices[layer_num]
                next_face_idx = face.connected_face_index[end_edge_idx]
                S = Segment(segment, face.idx, next_face_idx, end_vertex)
                sliced_layer.face_idx_to_seg_idx[face.idx] = len(sliced_layer.segments)
                sliced_layer.segments.append(S)
    return slices


def calculate_segment(p0, p1, p2, z):
    """
    """
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


class Slice(object):
    """
    A slice of a build, containing all segments within that layer
    """
    def __init__(self,
                 segments=None,
                 layer_number=-1,
                 height=4800,
                 width=7200,
                 transform=np.eye(3)):
        self.filename = os.path.join('./output', "layer_{}.bmp".format(layer_number))
        if segments is None:
            self.segments = []
        else:
            self.segments = segments
        self.layer_number = layer_number
        self.height = height
        self.width = width
        self.transform = transform
        self.polygons = []
        self.face_idx_to_seg_idx = {}
        self.open_polylines = []

    def make_polygons(self):
        for seg_idx, seg in enumerate(self.segments):
            if not seg.added_to_polygon:
                self.make_basic_polygon_loop(seg, seg_idx)

        if self.open_polylines:
            self.polygons = self.layer_graph()
        # Clear the segment list for this layer as it is no longer useful
        # self.segments = []

    def make_basic_polygon_loop(self, seg, start_seg_idx):
        """
        Create polygons from segments within every slice
        """
        # Start the polygon with the first piece of the segment
        # polygon = [seg.segment[0]]
        polygon = []
        # Begin tracking the segment index
        seg_idx = start_seg_idx

        # As long as there are valid segments, loop through them
        while seg_idx != -1:
            # Add segment end to the polygon
            seg = self.segments[seg_idx]
            polygon.append(seg.segment[1])
            seg.added_to_polygon = True
            seg_idx = self.get_next_seg_idx(seg, start_seg_idx)
            # If the polygon closes, add it to the list of polygons and
            # return
            if seg_idx == start_seg_idx:
                self.polygons.append(polygon)
                return

        self.open_polylines.append(polygon)
        # sliced_layer.polygons.append(polygon)
        return

    def get_next_seg_idx(self, seg, start_seg_idx):
        """
        Get the next segment idx to add to a polygon
        Utilizes the next face index that was calculated at
        the loading of the mesh
        """
        next_seg_idx = -1

        # If the segment end vertex is None, the segment ended at an edge
        if seg.end_vertex is None:
            if seg.next_face_idx != -1:
                return self.try_next_face_seg_idx(seg, seg.next_face_idx, start_seg_idx)
            else:
                return -1
        else:
            # if the segment ended at a vertex, look for other faces to try to get the
            # next segment
            for face in seg.end_vertex.connected_faces:
                result_seg_idx = self.try_next_face_seg_idx(seg, seg.face_idx, start_seg_idx)

                if result_seg_idx == start_seg_idx:
                    return start_seg_idx
                elif result_seg_idx != -1:
                    next_seg_idx = result_seg_idx
        return next_seg_idx

    def try_next_face_seg_idx(self, segment, face_idx, start_seg_idx):
        """
        This function finds another face that will continue the given segment
        :param segment:
        :param seg_idx:
        :param start_seg_idx:
        :return:
        """
        seg_idx = self.face_idx_to_seg_idx[face_idx]

        if seg_idx == start_seg_idx:
            return start_seg_idx
        elif self.segments[seg_idx].added_to_polygon:
            return -1
        else:
            return seg_idx

    def connect_open_polylines(self):
        # Find all possible stitches
        # stitch_pq = self.find_possible_stitches()
        pass

    def layer_graph(self):
        """
        This method uses an inefficient digraph approach to generate polygons.
        Can be used when there are open polylines that need to be joined.

        A more efficient method that can be used to join polylines can be
        found within the cura implementation.

        Make a digraph with edges representing from all segments
        Remove the bridges from digraph
        Return the cycles (aka polygons) of the digraph with all bridges removed.

        Useful links:
        - https://stackoverflow.com/questions/48736396/algorithm-to-find-bridges-from-cut-vertices
        - https://visualgo.net/en/dfsbfs
        :return:
        """
        digraph = nx.DiGraph()
        segs = [s.segment for s in self.segments]
        for seg in np.round(segs, decimals=3):
            p1 = tuple(seg[0])
            p2 = tuple(seg[1])

            if p1 == p2:
                pass
            else:
                digraph.add_edge(p1, p2)
        graph = nx.Graph(digraph)
        bridges = nx.bridges(graph)

        for bridge in bridges:
            try:
                digraph.remove_edge(bridge[0], bridge[1])
            except Exception:
                continue

        cycles = nx.simple_cycles(digraph)
        return list(cycles)

    def plot_polygons(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        for polygon_pts in self.polygons:
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
                try:
                    ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.05, length_includes_head=True)
                except:
                    pass
                    # print("problem at [x0, y0]: [{}, {}] and [x1, y1]: [{}, {}]".format(x0, y0, x1, y1))
        plt.show()

    def plot_segments(self, only_open_polylines=False):
        """
        Plots individual segments as separate colors
        """
        if only_open_polylines:
            segments = self.open_polylines
        else:
            segments = self.segments

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        for s in self.segments:
            seg = s.segment
            x = [seg[0][0], seg[1][0]]
            y = [seg[0][1], seg[1][1]]
            ax.plot(x, y, '.', lineStyle='None')

            x0 = seg[0][0]
            x1 = seg[1][0]
            y0 = seg[0][1]
            y1 = seg[1][1]
            ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.5, length_includes_head=True)
        plt.show()


class Segment(object):
    """
    A segment 
    """
    def __init__(self, segment, face_idx, next_face_idx, end_vertex):
        self.segment = segment
        self.face_idx = face_idx
        self.next_face_idx = next_face_idx
        self.added_to_polygon = False
        self.end_vertex = end_vertex


def write_layer(layer):
    layer.make_polygons()
    cv2_rasterize(layer.polygons, layer.filename, layer.layer_number, layer.height, layer.width)


def main():
    # f = './test_stl/logo.stl'
    f = './test_stl/q01.stl'
    # f = './test_stl/cylinder.stl'
    # f = './test_stl/prism.stl'
    # f = './test_stl/nist.stl'
    # f = './test_stl/hollow_prism.stl'
    # f = './test_stl/10_side_hollow_prism.stl'
    # f = './test_stl/concentric_1.stl'
    # f = './test_stl/links.stl'
    # f = './test_stl/square_cylinder.stl'
    # f = './test_stl/prism_hole.stl'
    # f = './test_stl/holey_prism.stl'
    optimized_mesh = OptimizedMesh(f)
    optimized_mesh.complete()
    resolution = 0.05
    Slices = slice_mesh(optimized_mesh, resolution)
    pool = Pool(5)
    pool.map(write_layer, Slices)
    # for layer in Slices:
    #     layer.make_polygons()
        # print("Layer Number: {}".format(layer.layer_number))
        # print("Polygons: {}".format(layer.polygons))
        # layer.plot_segments()
        # layer.plot_polygons()
        # layer.plot_segments()
        # if layer.open_polylines:
        #     layer.plot_segments(True)
        #     print(layer.open_polylines)
        # cv2_rasterize(layer.polygons, layer.filename, layer.layer_number, layer.height, layer.width)


if __name__ == '__main__':
    import cProfile
    cProfile.runctx('main()', globals(), locals(), filename=None)
    # main()
    # plt.arrow(0,0,2,2, head_width=0.05, length_includes_head=True)
    # plt.show()
