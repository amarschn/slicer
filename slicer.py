import stl
import numpy as np
import matplotlib.pyplot as plt
from image_writer import cv2_rasterize
import os
import sys
from multiprocessing import Pool
import ipdb


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
    # slice_list = []
    for i, layer in enumerate(layers):
        sl = Slice(layer_number=i)
        slices.append(sl)
        # slice_list.append([])

    for face in optimized_mesh.faces:
        p0 = face.vertices[face.vertex_indices[0]].p
        p1 = face.vertices[face.vertex_indices[1]].p
        p2 = face.vertices[face.vertex_indices[2]].p

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
                    end_vertex = face.vertices[face.vertex_indices[1]]

            elif z0 > z and z1 < z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p0, p1, p2, z)
                end_edge_idx = 2

            elif z0 >= z and z1 < z and z2 >= z:
                # What condition is this?
                segment = calculate_segment(p1, p0, p2, z)
                end_edge_idx = 1
                if p2[2] == z:
                    end_vertex = face.vertices[face.vertex_indices[2]]

            elif z0 < z and z1 > z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p1, p2, p0, z)
                end_edge_idx = 0

            elif z0 >= z and z1 >= z and z2 < z:
                # What condition is this?
                segment = calculate_segment(p2, p1, p0, z)
                end_edge_idx = 2
                if p0[2] == z:
                    end_vertex = face.vertices[face.vertex_indices[0]]

            elif z0 < z and z1 < z and z2 > z:
                # What condition is this?
                segment = calculate_segment(p2, p0, p1, z)
                end_edge_idx = 1

            else:
                # Not all cases create a segment
                continue

            if segment:
                sliced_layer = slices[layer_num]
                next_face = face.connected_face_index[end_edge_idx]
                S = Segment(segment, face.idx, next_face, end_vertex)
                sliced_layer.face_idx_to_seg_idx[face.idx] = len(sliced_layer.segments)
                sliced_layer.add_segment(S)
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
        self.filename = os.path.join('./output', "layer_{}.png".format(layer_number))
        if segments is None:
            self.segments = []
        else:
            self.segments = segments
        self.layer_number = layer_number
        self.height = 4800
        self.width = 7600
        self.transform = transform
        self.polygons = []
        self.face_idx_to_seg_idx = {}

    def add_segment(self, segment):
        self.segments.append(segment)

    def make_polygons(self, sliced_layer):
        for seg_idx, seg in enumerate(sliced_layer.segments):
            if not seg.added_to_polygon:
                self.make_basic_polygon_loop(sliced_layer, seg, seg_idx)

        sliced_layer.segments = []

    def make_basic_polygon_loop(self, sliced_layer, seg, start_seg_idx):
        """
        Create polygons from segments within every slice
        """
        # Start the polygon with the first piece of the segment
        polygon = [seg[0]]
        # Begin tracking the segment index
        seg_idx = start_seg_idx

        # As long as there are valid segments, loop through them
        while seg_idx != -1:
            # Add segment end to the polygon
            seg = sliced_layer.segments[seg_idx]
            polygon.append(seg[1])
            seg.added_to_polygon = True
            seg_idx = self.get_next_seg_idx(seg, start_seg_idx)
            # If the polygon closes, add it to the list of polygons and
            # return
            if seg_idx == start_seg_idx:
                sliced_layer.polygons.append(polygon)
                return

        # TODO: Need to handle open polylines...?
        # sliced_layer.polygons.append(polygon)
        return

    def get_next_seg_idx(self, seg, start_seg_idx):
        """
        Get the next segment idx to add to a polygon
        Utilizes the next face index that was calculated at
        the loading of the mesh
        """
        next_seg_idx = -1

        if seg.end_vertex:
            if seg.next_face_idx != -1:
                return self.try_face_next_seg_idx(seg, seg.next_face_idx, start_seg_idx)
        else:
            # if the segment ended at a vertex, look for other faces to try to get the
            # next segment
            for face in seg.end_vertex.connected_faces:
                result_seg_idx = self.try_next_face_seg_idx(seg, face.idx, start_seg_idx)

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
    mesh = stl.Mesh.from_file(f)
    resolution = 1.0
    Slices = get_slices(mesh, resolution)
    # for Slice in Slices:
    #     cv2_raserize(Slice)
    # pool = Pool(5)
    # pool.map(write_layer, Slices)



if __name__ == '__main__':
    import cProfile
    cProfile.runctx('main()', globals(), locals(), filename=None)
    # main()
    # plt.arrow(0,0,2,2, head_width=0.05, length_includes_head=True)
    # plt.show()
