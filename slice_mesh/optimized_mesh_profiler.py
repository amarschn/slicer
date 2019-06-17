"""Optimized mesh profiling function.

Impossible Objects
Drew Marschner
6/12/19
"""


if __name__ == '__main__':
    import pstats
    import cProfile
    # import pyximport
    # pyximport.install()

    import optimized_mesh_old
    f = '../test_stl/links.stl'
    cProfile.runctx("optimized_mesh_old.OptimizedMesh(f)", globals(), locals(),
                    "optimized_mesh.prof")
    s = pstats.Stats("optimized_mesh.prof")
    s.strip_dirs().sort_stats("time").print_stats()
