"""
Extracts unordered segments from STL files. Interesting but not as fast as other methods.
"""

import trimesh
import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt


def plot_polygon_from_points(polygon_pts):
    """
    Given an Nx2 array, plots polygons from these points, with arrows from one vertex to the next
    :param points:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
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


def main():
    resolution = 0.05
    mesh = trimesh.load_mesh('./test_stl/nist.stl')
    z_extents = mesh.bounds[:, 2]
    z_levels = np.arange(*z_extents, step=resolution)
    sections = mesh.section_multiplane(plane_origin=mesh.bounds[0], plane_normal=[0, 0, 1], heights=z_levels)
    # for section in sections:
    #     plot_polygon_from_points(section.vertices)
    #     print(section.vertices)
    # section = sections[1]
    # section.show()

if __name__ == "__main__":
    import cProfile
    cProfile.runctx('main()', globals(), locals(), filename=None)
    # main()