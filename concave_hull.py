"""


Source: http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
"""
import shapely
from shapely.ops import cascaded_union, polygonize
import scipy.spatial
import numpy as np
import math
import ipdb


def concave_hull(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: list of (x,y) tuples
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    points = tuple(map(tuple, points))
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return shapely.geometry.MultiPoint(points).convex_hull

    def add_edge(edges, edge_points, p1, p2):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (p1, p2) in edges or (p2, p1) in edges:
            # already added
            return
        edges.add((p1, p2))
        edge_points.append((points[p1], points[p2]))
        # edge_points.append(points[p1])
        # edge_points.append(points[p2])

    tri = scipy.spatial.Delaunay(points)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
    	# ipdb.set_trace()
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Lengths of sides of triangle
        len_a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        len_b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        len_c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (len_a + len_b + len_c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-len_a)*(s-len_b)*(s-len_c))
        circum_r = len_a*len_b*len_c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, ia, ib)
            add_edge(edges, edge_points, ib, ic)
            add_edge(edges, edge_points, ic, ia)
    m = shapely.geometry.MultiLineString(edge_points)

    triangles = list(polygonize(m))

    return cascaded_union(triangles), edge_points
    # return edge_points

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	POINTS = np.loadtxt('h_pts.csv')
	hull, _ = concave_hull(POINTS, alpha=0.06)
	points = []
	try:
		for poly in hull:
			points.extend(list(poly.exterior.coords))
	except:
		points = list(hull.exterior.coords)
	# ipdb.set_trace()
	# points = list(hull[0].exterior.coords)
	# for poly in 

	# edge_points = concave_hull(POINTS, alpha=0.5)
	
	plt.plot(*zip(*points), marker='o', linestyle=None)
	plt.show()

