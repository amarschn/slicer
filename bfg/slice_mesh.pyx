import numpy as np
import networkx as nx


class Slice(object):
    """
    The Slice object is used to represent a single slice of a build, and once it is given the segments of that slice,
    can be used to construct the polygons of a slice.
    """
    def __init__(self, layer_number=-1, settings=None):
        """
        :param layer_number: The layer number of this particularly slice
        :param settings: A dict containing the settings of the build
        """
        self.segments = []
        self.layer_number = layer_number
        self.polygons = []
        self.face_idx_to_seg_idx = {}
        self.open_polylines = []
        self.settings = settings

    def make_polygons(self):
        """
        Creates polygons from a list of segments
        :return: None
        """
        for seg_idx, seg in enumerate(self.segments):
            if not seg.added_to_polygon:
                self.make_basic_polygon_loop(seg, seg_idx)

        # If some of the polylines are open, it means the optimized mesh data structure was insufficient. If this occurs
        # the layer_graph function is used to analyze the segments directly. This has a time penalty but is more likely
        # to result in fully defined polygons.
        # TODO: implement Cura's method of open polygon fixing (it is complicated but faster than the digraph approach)
        if self.open_polylines:
            self.polygons = self.layer_graph()

        # Clear the segment list for this layer as it is no longer useful
        self.segments = []

    def make_basic_polygon_loop(self, seg, start_seg_idx):
        """
        Create polygons from segments within every slice
        :param seg: an array containing the start and end point of the segment with which to start the polygon
        :param start_seg_idx: the index of the segment that is the first argument
        """
        # Start the polygon with the first piece of the segment
        polygon = [seg.segment[0]]

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
        return

    def get_next_seg_idx(self, seg, start_seg_idx):
        """
        Get the next segment idx to add to a polygon
        Utilizes the next face index that was calculated at the loading of the mesh
        :param seg: an array containing the start and end point of the segment with which to start the polygon
        :param start_seg_idx: the index of the starting segment of the polygon being constructed
        :return next_seg_idx: the index of the next segment to add to the polygon, or -1
        """
        next_seg_idx = -1

        # If the segment end vertex is None, the segment ended at an edge
        if seg.end_vertex is None:
            if seg.next_face_idx != -1:
                return self.try_next_face_seg_idx(seg.next_face_idx, start_seg_idx)
            else:
                return -1
        else:
            # if the segment ended at a vertex, look for other faces to try to get the
            # next segment
            for face in seg.end_vertex.connected_faces:
                if face != seg.face_idx:
                    result_seg_idx = self.try_next_face_seg_idx(face, start_seg_idx)

                    if result_seg_idx == start_seg_idx:
                        return start_seg_idx
                    elif result_seg_idx != -1:
                        next_seg_idx = result_seg_idx
        return next_seg_idx

    def try_next_face_seg_idx(self, face_idx, start_seg_idx):
        """
        This function finds another face that will continue the given segment
        :param face_idx: the index of the face that is being used to attempt to find a segment that could be added to
                         the polygon under construction
        :param start_seg_idx: the index of the starting segment of the polygon being constructed
        :return: the next segment to be considered for adding to the polygon
        """
        try:
            seg_idx = self.face_idx_to_seg_idx[face_idx]
        except KeyError:
            return -1

        if seg_idx == start_seg_idx:
            return start_seg_idx
        if self.segments[seg_idx].added_to_polygon:
            return -1
        else:
            return seg_idx


    def layer_graph(self):
        """
        This method uses an inefficient digraph approach to generate polygons.
        Can be used when there are open polylines that need to be joined.

        Process:
        --------
        Make a digraph with edges representing from all segments
        Remove the bridges from digraph
        Return the cycles (aka polygons) of the digraph with all bridges removed.

        Useful links:
        - https://stackoverflow.com/questions/48736396/algorithm-to-find-bridges-from-cut-vertices
        - https://visualgo.net/en/dfsbfs
        :return: an array containing the polygons of the slice
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


class Segment(object):
    """
    A class representation of a 2D segment within a slice, with added information related to the position of the
    segment within the slice w.r.t the faces of the mesh. This added information is useful when constructing polygons
    from the list of segments within a slice.
    """
    def __init__(self, segment, face_idx, next_face_idx, end_vertex):
        """

        :param segment: An array containing the points of the segment [[x0,y0],[x1,y1]]
        :param face_idx: The index of the face that is being sliced to create the segment.
        :param next_face_idx: The index of the neighboring face of the segment's end point
        :param end_vertex: The end vertex of the segment, if it has one. Segments only have an end_vertex if the end
                           of the segment is at the vertex of a face of the mesh.
        """
        self.segment = segment
        self.face_idx = face_idx
        self.next_face_idx = next_face_idx
        self.added_to_polygon = False
        self.end_vertex = end_vertex


def slice_mesh(optimized_mesh, settings):
    """
    This function "slices" an optimized mesh object, by going through the following process:

    For each face in the mesh:
        For each layer in the number of layers that will be created:
            create a segment which is the intersection of the face and the layer plane
            add the segment to the Slice object, which will contain all of the segments of a given layer

    :param optimized_mesh: an object that contains all of the needed mesh information for slicing.
    :param settings: dict of build settings.
    :return: array of Slice objects, each slice object representing a slice of the build.
    """
    resolution = settings["layer_height"]
    height = optimized_mesh.mesh.z.max() - optimized_mesh.mesh.z.min()
    layers = np.array([z for z in range(int(height / resolution) + 1)]) * resolution

    slices = []

    for i, layer in enumerate(layers):
        sl = Slice(layer_number=i, settings=settings)
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
                segment = calculate_segment(p0, p2, p1, z)
                end_edge_idx = 0
                if p1[2] == z:
                    end_vertex = optimized_mesh.vertices[face.vertex_indices[1]]

            elif z0 > z and z1 < z and z2 < z:
                segment = calculate_segment(p0, p1, p2, z)
                end_edge_idx = 2

            elif z0 >= z and z1 < z and z2 >= z:
                segment = calculate_segment(p1, p0, p2, z)
                end_edge_idx = 1
                if p2[2] == z:
                    end_vertex = optimized_mesh.vertices[face.vertex_indices[2]]

            elif z0 < z and z1 > z and z2 < z:
                segment = calculate_segment(p1, p2, p0, z)
                end_edge_idx = 0

            elif z0 >= z and z1 >= z and z2 < z:
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


cdef double interpolate(float y, float y0, float y1, float x0, float x1):
    """Interpolates an x value to match the y value, given a line with start point at x0,y0 and end point at x1, y1
    This function is kind of confusing, it is often helpful to draw it out by drawing a straight angled line, and
    a straight horizontal line intersecting it, and using interpolation to find the intersection point.
    :param y: y-value at which the intersected line is being intersected. We want to know the x-value of this line.
    :param y0: intersected line beginning y-point
    :param y1: intersected line end y-point
    :param x0: intersected line beginning x-point
    :param x1: intersected line end x-point
    """
    # TODO: should these be ints instead for speed reasons?
    cdef float dx = x1 - x0
    cdef float dy = 1/(y1 - y0)
    cdef float p
    cdef float x
    # If the slope is negative
    if dy < 0:
        # the proportion of the curve we are interpolating when the slope is negative is flipped
        p = (y - max(y0, y1)) * dy
        x = dx * p + x0
    else:
        p = (y - min(y0, y1)) * dy
        x = dx * p + x0
    return x


cdef calculate_segment(float[:] p0, float[:] p1, float[:] p2, float z):
    """Calculates a segment.
    :param p0: float array of the first 3 points of a triangle
    :param p1: float array of the second 3 points of a triangle
    :param p2: float array of the third 3 points of a triangle
    :param z: float value of the height at which to "slice" the triangle
    """
    cdef float x_start, x_end, y_start, y_end

    x_start = interpolate(z, p0[2], p1[2], p0[0], p1[0])
    x_end = interpolate(z, p0[2], p2[2], p0[0], p2[0])

    y_start = interpolate(z, p0[2], p1[2], p0[1], p1[1])
    y_end = interpolate(z, p0[2], p2[2], p0[1], p2[1])

    return [(x_start, y_start), (x_end, y_end)]