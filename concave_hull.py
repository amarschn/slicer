"""
Drew Marschner

TODO:

1. Smart subsampling of layer points - get an accurate approximation of the
   part.
2. Create a test suite - test multiple different STL types
	- Single simple part
	- Complex simple part
	- Simple multiple parts
	- Complex multiple parts
3. Fix invalid hull issue
"""


from scipy.spatial import cKDTree
import shapely.geometry
import numpy as np
import math
import ipdb


def concave_hull(points, k):
	"""
	Paper: "CONCAVE HULL: A K-NEAREST NEIGHBOURS APPROACH FOR THE COMPUTATION OF THE REGION OCCUPIED BY A SET OF POINTS"
	Implementation: https://github.com/joaofig/uk-accidents/blob/master/geomath/hulls.py
	https://pdfs.semanticscholar.org/2397/17005c3ebd5d6a42fc833daf97a0edee1ce4.pdf
	points: 2D array of XY points in the format [(x1,y1), (x2,y2), ...]
	k: number of NN to use in algorithm
	"""
	
	kk = max(k, 3)
	points = set(map(tuple, points))
	if len(points) < 3:
		return []
	if len(points) == 3:
		return points
	kk = min(kk, len(points) - 1)

	# Construct a KD tree for use in knn search
	tree = cKDTree(np.array(list(points)), copy_data=True, balanced_tree=True)

	# Set first point to the minimum y-value (why y-value?)
	# ipdb.set_trace()
	first = min(points, key=lambda pt: pt[1])
	hull = np.array([first])
	current = first
	points.remove(first)
	# Set previous to a point slightly to the right of the current point (angle of 0deg)
	previous = (current[0] + 1, current[1])
	step = 2

	while ((current != first or step == 2) and (len(points) > 0)):
		# At step 5 it is possible to create a >3-sided polygon
		if step == 5:
			points.add(first)
		# ipdb.set_trace()
		nearest_pts = knn(tree, k + 1, current, previous)
		candidates = rank_points(nearest_pts, current, previous)
		# ipdb.set_trace()
		i = 0
		invalid_hull = True

		while invalid_hull and i < len(candidates):
			candidate = candidates[i]
			c = tuple(candidate)

			# Create a test hull to check if there is any self interesection
			test_hull = hull[:]
			# ipdb.set_trace()
			test_hull = np.append(test_hull, [candidate], axis=0)
			line = shapely.geometry.asLineString(test_hull)
			invalid_hull = not line.is_simple
			i += 1

		# If all points still intersect, it means we need to get new points to examine
		if invalid_hull:
			# ipdb.set_trace()
			print candidates
			print current
			print "invalid hull"
			return hull
		previous = current
		current = tuple(candidate)
		# ipdb.set_trace()
		points.remove(current)
		hull = test_hull
		# print hull
		step += 1
	return hull


def knn(kdtree, k, current, previous):
	"""
	Gets k nearest neighbors from kdtree, removing currenet and previous points
	if necessary
	"""
	_, nearest_pts_idx = kdtree.query(current, k=k)
	nearest_pts = kdtree.data[nearest_pts_idx]
	filtered_pts = []
	# ipdb.set_trace()
	for pt in nearest_pts:
		if not all(pt == current) and not all(pt == previous):
			filtered_pts.append(pt)

	return np.array(filtered_pts)




def rank_points(pts, current, previous):
	"""
	Ranks points according to 
	"""
	angles = np.array([])
	ordered_points = np.array([])
	# ipdb.set_trace()
	v1 = (current[0] - previous[0], current[1] - previous[1])
	for pt in pts:
		v2 = (pt[0] - current[0], pt[1] - current[1])
		angles = np.append(angles, get_angle(v1, v2))
	# ipdb.set_trace()
	return pts[np.argsort(angles)[::-1]]


def get_angle(v1, v2):
	"""
	Returns angle between 2 vectors
	"""
	(x1, y1) = v1
	(x2, y2) = v2
	return math.atan2(x1*y2 - y1*x2, x1*x2 + y1*y2)




if __name__ == '__main__':
	# pts = [(-1,0.1), (0,1), (1,0)]
	# current = (0,0)
	# previous = (-1,0)
	# print rank_points(pts, current, previous)
	import matplotlib.pyplot as plt
	POINTS = np.loadtxt('h_pts.csv')

	# Thin the points
	# POINTS = POINTS[::500]

	# POINTS = np.loadtxt('bracket1.csv')
	# POINTS = np.loadtxt('h_pts.csv')
	# POINTS = np.loadtxt('test_pts.csv')
	# hull, _ = concave_hull(POINTS, alpha=0.06)
	print(len(POINTS))
	hull = concave_hull(POINTS, k=10)
	# points = []
	# try:
	# 	for poly in hull:
	# 		points.extend(list(poly.exterior.coords))
	# except:
	# 	points = list(hull.exterior.coords)
	# ipdb.set_trace()
	# points = list(hull[0].exterior.coords)
	# for poly in 

	# edge_points = concave_hull(POINTS, alpha=0.5)
	plt.plot(*zip(*POINTS), color='r', marker='x', linestyle='None')
	plt.plot(*zip(*hull), marker='o', linestyle='None')
	plt.show()

