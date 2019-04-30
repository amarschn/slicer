import pickle
import numpy as np
import matplotlib.pyplot as plt

ALL_SEGMENTS = './failed_layers/graph_failure_layer_3'
BROKEN_SEGMENTS = './failed_layers/broken_segments'

with open(ALL_SEGMENTS, 'rb') as f:
    all_segments = pickle.load(f)

with open(BROKEN_SEGMENTS, 'rb') as f:
    broken_segments = pickle.load(f)

all_segments = np.array(all_segments)
broken_segments = np.array(broken_segments)

all_segments_x = all_segments[:,0,0]
all_segments_y = all_segments[:,0,1]

broken_segments_x = broken_segments[:,0,0]
broken_segments_y = broken_segments[:,0,1]

plt.plot(all_segments_x, all_segments_y, 'rx')
plt.plot(broken_segments_x, broken_segments_y, 'bo')
plt.show()


