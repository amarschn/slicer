#cython: profile=True

import stl
import numpy as np
from cpython cimport array
import array

DECIMALS = 3

class MeshFace(object):
    def __init__(self, idx, vertex_indices):
        self.idx = idx
        self.vertex_indices = vertex_indices
        # The connected face index will have the same ordering as the
        # vertex indices, meaning connected face index 0 is connected
        # via vertex index 0 and 1, etc.
        self.connected_face_index = array.array('i', [])
        self.has_disconnected_faces = 0

class Vertex(object):

    def __init__(self, idx, p):
        self.idx = idx
        self.connected_faces = []
        self.p = p


class OptimizedMesh(object):
    """
    Modeled on CuraEngine's optimized model of a mesh, links neighboring
    faces so that slicing can be much faster

    The vertex hash map is a dictionary that contains:

    key (hash) : [array of indices representing vertices]

    """

    def __init__(self, file):
        self.mesh = stl.Mesh.from_file(file)
        self.triangles = np.round(self.mesh.vectors, decimals=DECIMALS)
        self.vertex_hash_map = {}
        self.vertices = []
        self.faces = []
        self.add_faces()

    def add_faces(self):
        for triangle in self.triangles:
            v0 = triangle[0]
            v1 = triangle[1]
            v2 = triangle[2]

            vi0 = self.find_idx_of_vertex(v0)
            vi1 = self.find_idx_of_vertex(v1)
            vi2 = self.find_idx_of_vertex(v2)

            if vi0 == vi1 or vi1 == vi2 or vi0 == vi2:
                continue

            face_idx = len(self.faces)
            f = MeshFace(face_idx, (vi0, vi1, vi2))
            self.faces.append(f)

            self.vertices[vi0].connected_faces.append(face_idx)
            self.vertices[vi1].connected_faces.append(face_idx)
            self.vertices[vi2].connected_faces.append(face_idx)

    def find_idx_of_vertex(self, v):
        """
        Returns the index of a vertex.
        If the vertex is not in the vertex hash map, it is added.
        """
        # Find the hash of the vertex
        v_hash = tuple(v)

        # If the vertex hash is already stored, then get the key
        index = self.vertex_hash_map.get(v_hash)

        if index is not None:
            return index
        else:
            index = len(self.vertices)
            self.vertex_hash_map[v_hash] = index
            vertex = Vertex(index, v)
            self.vertices.append(vertex)
            return index


    def get_face_idx_with_points(self, idx0, idx1, not_face_idx, not_vertex_idx):
        """
        Returns the index of the other face connected to the edge between
        vertices with indices idx0 and idx1

        If there are more than two faces connected by the edge, then the next
        is considered the next counter-clockwise face, ordered from idx1 to
        idx0

        idx0 : the first vertex index
        idx1 : the second vertex index
        not_face_idx : the index of a face which should not be returned
        not_vertex_idx : third vertex of the face not_face_idx
        """
        candidate_faces = []

        for connected_face in self.vertices[idx0].connected_faces:
            if connected_face == not_face_idx:
                continue
            if ((self.faces[connected_face].vertex_indices[0] == idx1) or
                (self.faces[connected_face].vertex_indices[1] == idx1) or
                (self.faces[connected_face].vertex_indices[2] == idx1)):
                candidate_faces.append(connected_face)

        if not candidate_faces:
            # print("Unconnected faces, bad mesh!")
            self.has_disconnected_faces = True
            return -1

        if len(candidate_faces) == 1:
            return candidate_faces[0]

        if len(candidate_faces) % 2 == 0:
            # print("Edge with uneven number of faces connecting it")
            self.has_disconnected_faces = True

        ##############################
        # MATH STUFF - HERE BE DRAGONS
        # - https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
        ##############################

        # Calculate the vector of the edge
        v0 = self.vertices[idx1].p - self.vertices[idx0].p
        # Calculate the normal of the plane in which the normals of the faces connected to the edge lie
        # Except actually this appears to just be the unit vector of the edge
        norm = np.linalg.norm(v0)
        if norm != 0:
            n = v0 / norm
        else:
            n = v0

        #
        other_edge = self.vertices[not_vertex_idx].p - self.vertices[idx0].p
        n0 = np.cross(other_edge, v0)

        smallest_angle = 1000 # Make the smallest angle bigger than the biggest possible angle (2pi)
        best_idx = -1

        for candidate_face in candidate_faces:
            # Look through the vertices of the candidate face, find the idx of the vertex that
            # is not idx0 or idx1
            idx2 = [idx for idx in candidate_face.vertex_indices if idx != idx0 and idx != idx1][0]
            v1 = self.vertices[idx2].p - self.vertices[idx0].p
            n1 = np.cross(v0, v1)

            dot = n0 * n1
            det = n * np.cross(n0, n1)
            angle = np.arctan2(det, dot)

            if angle < 0:
                angle += 2*np.pi

            if angle == 0:
                print("Mesh has overlapping faces")
                pass

            if angle < smallest_angle:
                smallest_angle = angle
                best_idx = candidate_face

        if best_idx == -1:
            # print("Mesh has disconected faces")
            self.has_disconnected_faces = True


        return best_idx

    def complete(self):
        self.vertex_hash_map = {}
        for i, face in enumerate(self.faces):
            # print(i)
            face.connected_face_index.append(self.get_face_idx_with_points(face.vertex_indices[0], face.vertex_indices[1], i, face.vertex_indices[2]))
            face.connected_face_index.append(self.get_face_idx_with_points(face.vertex_indices[1], face.vertex_indices[2], i, face.vertex_indices[0]))
            face.connected_face_index.append(self.get_face_idx_with_points(face.vertex_indices[2], face.vertex_indices[0], i, face.vertex_indices[1]))

if __name__ == '__main__':
    # f = './test_stl/4_parts.stl'
    f = './test_stl/q01.stl'
    o = OptimizedMesh(f)
    o.complete()

