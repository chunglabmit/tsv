import contextlib
import glob
import numpy as np
import os
import unittest
import tifffile
import tempfile
import shutil
from tsv.downsample import main


@contextlib.contextmanager
def make_case(shape):
    """
    Make a stack of files

    :param shape: the shape of the TIF files to make
    :return: a 3 tuple of source glob expr for the files, destination directory
    for the downsampled files and a 3D stack of tif files.
    """
    stack = np.random.RandomState(1234).randint(0, 65535, shape)\
        .astype(np.uint16)
    src = tempfile.mkdtemp()
    for i, plane in enumerate(stack):
        tifffile.imsave(os.path.join(src, "img_%04d.tiff" % i), plane)
    dest = tempfile.mkdtemp()
    yield(
        os.path.join(src, "img_*.tiff"),
        dest,
        stack
    )
    try:
        shutil.rmtree(src)
    except:
        print("Warning: failed to remove %s" % src)
    try:
        shutil.rmtree(dest)
    except:
        print("Warning: failed to remove %s" % dest)


class TestDownsample(unittest.TestCase):
    def test_downsample_2(self):
        with make_case((20, 64, 64)) as (src_glob, dest, stack):
            main(["--src", src_glob, "--dest", dest])
        dest_files = sorted(glob.glob(
            os.path.join(dest, os.path.split(src_glob)[1])))
        for dest_file, plane in zip(dest_files, stack):
            img = tifffile.imread(dest_file)
            self.assertSequenceEqual(img.shape, (32, 32))

    def test_downsample_4(self):
        with make_case((20, 64, 64)) as (src_glob, dest, stack):
            main(["--src", src_glob, "--dest", dest,
                  "--downsample-factor", "4"])
        dest_files = sorted(glob.glob(
            os.path.join(dest, os.path.split(src_glob)[1])))
        for dest_file, plane in zip(dest_files, stack):
            img = tifffile.imread(dest_file)
            self.assertSequenceEqual(img.shape, (16, 16))


if __name__ == '__main__':
    unittest.main()
