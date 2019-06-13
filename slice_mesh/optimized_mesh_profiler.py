"""Optimized mesh profiling function.

Impossible Objects
Drew Marschner
6/12/19
"""


def create_optimized_mesh(f):
    """Test function."""
    mesh = optimized_mesh.OptimizedMesh(f)
    # mesh.complete()
    return mesh


if __name__ == '__main__':
    import pstats
    import cProfile
    import pyximport
    pyximport.install()

    import optimized_mesh

    f = '../test_stl/links.stl'
    cProfile.runctx("optimized_mesh.OptimizedMesh(f)", globals(), locals(),
                    "optimized_mesh.prof")
    s = pstats.Stats("optimized_mesh.prof")
    s.strip_dirs().sort_stats("time").print_stats()
