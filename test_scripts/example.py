import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import slicer

def main():
    # f = './test_stl/logo.stl'
    # f = './test_stl/q01.stl'
    # f = './test_stl/cylinder.stl'
    f = './test_stl/prism.stl'
    # f = './test_stl/nist.stl'
    # f = './test_stl/hollow_prism.stl'
    # f = './test_stl/10_side_hollow_prism.stl'
    # f = './test_stl/concentric_1.stl'
    # f = './test_stl/links.stl'
    # f = './test_stl/square_cylinder.stl'
    # f = './test_stl/prism_hole.stl'
    # f = './test_stl/holey_prism.stl'

    s = slicer.slicer.Slicer(f)
    s.create_images()

if __name__ == '__main__':
    import cProfile
    cProfile.runctx('main()', globals(), locals(), filename=None)