import contextlib
import numpy as np
import os
import shutil
import tempfile
import tifffile
import unittest

from tsv.simple import main

@contextlib.contextmanager
def make_case(xs, ys, zs, tile_shape):
    """Make a test case

    :param xs: the starting x coordinates of the tiles in microns
    :param ys: the starting y coordinates of the tiles
    :param zs: the starting z coordinates of the tiles
    :param tile_shape: the shape in pixels of each tile (y, x)
    :returns: the root directory of the tiles
    """
    r = np.random.RandomState(1234)
    root = tempfile.mkdtemp()
    dest = tempfile.mkdtemp()
    for x in xs:
        xdir = os.path.join(root, str(x))
        os.mkdir(xdir)
        for y in ys:
            ydir = os.path.join(xdir, "%d_%d" % (x, y))
            os.mkdir(ydir)
            for z in zs:
                path = os.path.join(ydir, "%06d.tif" % z)
                tile = r.randint(0, 4095, tile_shape).astype(np.uint16)
                tifffile.imsave(path, tile)
    yield root, dest
    shutil.rmtree(root)
    shutil.rmtree(dest)


class TestSimple(unittest.TestCase):
    def test_xy(self):
        with make_case(xs=[1000, 2000], ys=[2000, 3000], zs=[1, 2],
                       tile_shape=(100, 100)) as (src, dest):
            pattern = os.path.join(dest, "img_{z:04d}.tiff")
            main(["--path", src,
                  "--output-pattern", pattern,
                  "--voxel-size-xy", "2"])
            img_path = pattern.format(z=0)
            img = tifffile.imread(img_path)
            self.assertSequenceEqual(img.shape, (150, 150))

    def test_x_different_from_y(self):
        with make_case(xs=[1000, 2000], ys=[2000, 3000], zs=[1, 2],
                       tile_shape=(100, 100)) as (src, dest):
            pattern = os.path.join(dest, "img_{z:04d}.tiff")
            main(["--path", src,
                  "--output-pattern", pattern,
                  "--voxel-size-x", "2",
                  "--voxel-size-y", "1"])
            img_path = pattern.format(z=0)
            img = tifffile.imread(img_path)
            self.assertSequenceEqual(img.shape, (200, 150))


if __name__ == '__main__':
    unittest.main()
