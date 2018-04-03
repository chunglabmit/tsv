import numpy as np
import os
import subprocess
import tempfile
import tifffile
import unittest

from tsv.volume import VExtentBase, VExtent
from tsv import convert

class MockVolume(VExtentBase):
    def __init__(self, img):
        self.img = img

    @property
    def x0(self):
        return 0

    @property
    def x1(self):
        return self.img.shape[2]

    @property
    def y0(self):
        return 0

    @property
    def y1(self):
        return self.img.shape[1]

    @property
    def z0(self):
        return 0

    @property
    def z1(self):
        return self.img.shape[0]

    def imread(self, volume, dtype):
        result = self.img[volume.z0:volume.z1,
                          volume.y0:volume.y1,
                          volume.x0:volume.x1].astype(dtype)
        return result


class TestConvert(unittest.TestCase):
    def test_tsv_convert_2D_tif_command(self):
        subprocess.check_call(["tsv-convert-2D-tif", "--help"])

    def test_tsv_diag_stack(self):
        subprocess.check_call(["tsv-diag-stack", "--help"])

    def test_convert_one_plane(self):
        with tempfile.TemporaryDirectory() as td:
            output_pattern = os.path.join(td, "img_{z:04d}.tif")
            img = np.random.RandomState(1234).randint(0, 255, (20, 20, 20))
            mock = MockVolume(img)
            v = VExtent(1, 11, 2, 14, 3, 16)
            convert.convert_one_plane(mock, 3, 1, np.uint8,
                                      output_pattern, v, 5)
            expected_path = output_pattern.format(z=5)
            self.assertTrue(os.path.exists(expected_path))
            plane = tifffile.imread(expected_path)
            self.assertEqual(plane.dtype, np.dtype(np.uint8))
            np.testing.assert_equal(plane, img[5, 2:14, 1:11])

    def test_convert_one_plane_decimation(self):
        with tempfile.TemporaryDirectory() as td:
            output_pattern = os.path.join(td, "img_{z:04d}.tif")
            img = np.random.RandomState(1234).randint(0, 255, (20, 20, 20))
            mock = MockVolume(img)
            v = VExtent(1, 13, 2, 18, 3, 16)
            convert.convert_one_plane(mock, 3, 4, np.uint8,
                                      output_pattern, v, 5)
            expected_path = output_pattern.format(z=5)
            self.assertTrue(os.path.exists(expected_path))
            plane = tifffile.imread(expected_path)
            self.assertEqual(plane.dtype, np.dtype(np.uint8))
            np.testing.assert_equal(plane, img[5, 2:18:4, 1:13:4])


if __name__ == '__main__':
    unittest.main()
