import numpy as np
import ipdb


def polygon_orientation(polygon):
	"""
	TODO: this should be implemented in the initial polygon construction,
	      thus avoiding going over the list of points twice
	Calculated by determining the minimum y of the polygon, and then choose
	whichever minimum (if there is a tie) has the maximum x, this will
	guarantee that the point lies on the convex hull of the polygon. Once
	this minimum point is found calculate the sign of the cross product
	between the two segments that the point is a part of

	Returns a boolean
	True indicates a positive orientation "counter clockwise"
	False indicates a negative orientation "clockwise"
	"""
	min_y = polygon[0][0][1]
	min_y_x = polygon[0][0][0]

	points = np.array(polygon)[0,:,0]
	left_neighbor = points[-1]
	right_neighbor = points[1]

	for idx, pt in enumerate(points):
		x = pt[0]
		y = pt[1]

		if (y < min_y) or (y == min_y and x > min_y_x):
			min_y = y
			min_y_x = x
			if idx > 0:
				left_neighbor_idx = idx - 1
			else:
				left_neighbor_idx = len(points)
			if idx < len(points):
				right_neighbor_idx = idx + 1
			else:
				right_neighbor_idx = 0

	left_seg = [points[left_neighbor_idx], [min_y_x, min_y]]
	right_seg = [[min_y_x, min_y], [points[right_neighbor_idx]]]

	return np.sign(np.cross())



	for idx, segment in enumerate(polygon):
		for point in segment:
			if point[1] <= min_y:
				if point[0] <

		start_y = segment[0][1]
		end_y = segment[1][1]
		if start_y <= min_y:

			min_y = start_y

			seg1 = segment
			if idx == 0:
				neighbor = len(polygon)
			else:
				neighbor = idx - 1
			seg2 = polygon[neighbor]
		
	ipdb.set_trace()
	return 0





def layer_svg(polygons, layer, width=304.8, height=203.2):
	"""
	"""
	svg = """<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\" 
	xmlns:svg=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" 
	viewport-fill=\"black\">\n
	<!-- Generated using Drew's Slicer -->\n
	""".format(width, height)

	for polygon in polygons:

		if polygon_orientation(polygon):
			color = "black"
		else:
			color = "white"

		poly_str = "\n<polygon style=\"fill: {}\" points=\"".format(color)

		for segment in polygon:
			x_start, y_start = segment[0]
			x_end, y_end = segment[1]
			poly_str += "{},{} ".format(x_start,y_start,x_end,y_end)
		poly_str += "\"></polygon>"

		svg += poly_str
	svg += "</svg>"
	return svg


if __name__ == '__main__':
	import pickle
	with open ('nist1', 'rb') as fp:
		polygons = pickle.load(fp)

	svg = layer_svg(polygons, 1)
	with open('layer.svg', 'w') as f:
		f.write(svg)
