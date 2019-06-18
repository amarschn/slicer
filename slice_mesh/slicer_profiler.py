"""Slicer profiling functions.

Impossible Objects
Drew Marschner
6/12/19
"""
import optimized_mesh
import slice_mesh


def create_optimized_mesh(f):
    """Test function."""
    mesh = optimized_mesh.OptimizedMesh(f)
    mesh.complete()
    return mesh


def mesh_slice(f, resolution=0.05):
    """Mesh slicing function."""
    opt_mesh = create_optimized_mesh(f)
    slices = slice_mesh.slice_mesh(opt_mesh, resolution)
    return slices


if __name__ == '__main__':
    import pstats
    import cProfile

    f = '../test_stl/q01.stl'
    cProfile.runctx("mesh_slice(f)", globals(), locals(),
                    "mesh_slice.prof")
    s = pstats.Stats("mesh_slice.prof")
    s.strip_dirs().sort_stats("time").print_stats()
