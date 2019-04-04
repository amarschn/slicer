import numpy as np
import matplotlib.pyplot as plt


def intersect(segment, layer):
    """
    Returns intersect point of a line segment and a given layer height
    """
    # If the segment top or bottom intersects the layer, return that point
    if layer > segment[0][2] and layer > segment[1][2]:
        return []
    elif layer < segment[0][2] and layer < segment[1][2]:
        return []
    elif segment[0][2] == layer:
        return segment[0]
    elif segment[1][2] == layer:
        return segment[1]
    else:
        dx = segment[1][0] - segment[0][0]
        dy = segment[1][1] - segment[0][1]
        dz = segment[1][2] - segment[0][2]
        p = (layer - min(segment[0][2], segment[1][2]))/dz
        return (segment[0][0] + dx*p, segment[0][1] + dy*p)


segment = [[0.,0.,0.], [10.,10.,7.]]
layer = 7
print(intersect(segment, layer))






# triangle1 = np.array([[0., 0., 0.],
#                      [5., 0., 5.],
#                      [0., 0., 10.]])

# z1 = 0.
# z2 = 1.
# z3 = 5.0
# z4 = 10.0



# plt.plot(triangle1)
# plt.show()