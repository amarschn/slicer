
"""
Author: Drew Marschner

"""

import stl


def create_face_neighbor_dict(mesh):
	"""
	Create a dict of neighbor faces.
	Each mesh face (aka a triangle) should have a single neighbor
	These neighbors will be used later during the slicing algorithm
	to connect contours together for a given slice.
	"""
	neighbor_dict = {}
	vertex_dict = {}

	for triangle in mesh:
