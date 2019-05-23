import contextlib
import numpy as np
import os
import tifffile
from tsv.raw import raw_imsave
from tsv.fill_blanks import main
import shutil
import tempfile
import unittest


@contextlib.contextmanager
def make_case(plane_coords, shape, use_raw):
    """
    Make a test case, with files for the plane coordinates.
    :param plane_coords: three tuples of the x, y and z of the planes
    :param shape: the x / y shape of a plane
    :param use_raw: if True, write files out in raw format.
    """
    src = tempfile.mkdtemp()
    dest = tempfile.mkdtemp()
    img = np.zeros(shape, np.uint16)
    for x, y, z in plane_coords:
        src_path_x = os.path.join(src, "%08d" % x)
        dest_path_x = os.path.join(dest, "%08d" % x)
        if not os.path.exists(src_path_x):
            os.mkdir(src_path_x)
        if not os.path.exists(dest_path_x):
            os.mkdir(dest_path_x)
        src_path_y = os.path.join(src_path_x, "%08d_%08d" % (x, y))
        dest_path_y = os.path.join(dest_path_x, "%08d_%08d" % (x, y))
        if not os.path.exists(src_path_y):
            os.mkdir(src_path_y)
        if not os.path.exists(dest_path_y):
            os.mkdir(dest_path_y)
        if use_raw:
            src_path_z = os.path.join(src_path_y, "%08d.raw" % z)
            raw_imsave(src_path_z, img)
        else:
            src_path_z = os.path.join(src_path_y, "%08d.tif" % z)
            tifffile.imsave(src_path_z, img)
        dest_path_z = os.path.join(dest_path_y, "%08d.tif" % z)
        tifffile.imsave(dest_path_z, img)
    yield (src, dest)
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(dest, ignore_errors=True)


class TestFillBlanks(unittest.TestCase):
    def test_01_full_tif(self):
        case = ((0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1))
        with make_case(case, (64, 64), False) as (src, dest):
            main(["--src", src, "--dest", dest])

    def test_02_full_raw(self):
        case = ((0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1))
        with make_case(case, (64, 64), True) as (src, dest):
            main(["--src", src, "--dest", dest])

    def test_03_missing_file(self):
        case = ((0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1))
        with make_case(case, (64, 64), True) as (src, dest):
            missing = os.path.join(dest, "00000001", "00000001_00000000",
                                   "00000001.tif")
            self.assertFalse(os.path.exists(missing))

            main(["--src", src, "--dest", dest])
            img = tifffile.imread(missing)
            self.assertSequenceEqual(img.shape, (64, 64))

    def test_04_missing_stack(self):
        case = ((0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 1, 0),
                (1, 1, 1))
        with make_case(case, (64, 64), True) as (src, dest):
            missing = [os.path.join(dest, "00000001", "00000001_00000000",
                                    "%08d.tif" % idx) for idx in (0, 1)]
            for m in missing:
                self.assertFalse(os.path.exists(m))
            main(["--src", src, "--dest", dest])
            img = tifffile.imread(missing)
            for m in missing:
                self.assertTrue(os.path.exists(m))
