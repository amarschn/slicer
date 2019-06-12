import pstats, cProfile
import pyximport
pyximport.install()

import slice_mesh
import optimized_mesh

f = '../test_stl/q01.stl'
resolution = 0.1
mesh = optimized_mesh.OptimizedMesh(f)
mesh.complete()

cProfile.runctx("slice_mesh.slice_mesh(mesh, resolution)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()