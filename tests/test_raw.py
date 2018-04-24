import numpy as np
import os
import unittest
import tempfile
from tsv.raw import raw_imread, raw_imsave


class TestRaw(unittest.TestCase):
    def test_raw_le_imread(self):
        # width = 16, little-endian
        # height = 24, little-endian
        header = b"\x10\x00\x00\x00" + \
                 b"\x18\x00\x00\x00"
        img = np.random.RandomState(1234).randint(0, 4095, (24, 16))\
            .astype(np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".raw") as fd:
            fd.write(header)
            fd.write(img.data)
            fd.flush()
            result = raw_imread(fd.name)
            np.testing.assert_equal(img, result)

    def test_raw_be_imread(self):
        header = b"\x00\x00\x00\x10" + \
                 b"\x00\x00\x00\x18"
        img = np.random.RandomState(4567).randint(0, 4095, (24, 16))\
            .astype(">u2")
        with tempfile.NamedTemporaryFile(suffix=".raw") as fd:
            fd.write(header)
            fd.write(img.data)
            fd.flush()
            result = raw_imread(fd.name)
            np.testing.assert_equal(img, result)

    def test_raw_imwrite(self):
        header = b"\x10\x00\x00\x00" + \
                 b"\x18\x00\x00\x00"
        img = np.random.RandomState(89).randint(0, 4095, (24, 16))\
            .astype(np.uint16)
        path = tempfile.mktemp(".raw")
        try:
            raw_imsave(path, img)
            with open(path, "rb") as fd:
                test_header = fd.read(8)
                self.assertEqual(test_header, header)
                test_img_data = fd.read(24 * 16 * 2)
                test_img = np.frombuffer(test_img_data, np.uint16)\
                    .reshape(24, 16)
                np.testing.assert_equal(test_img, img)
        finally:
            os.remove(path)

if __name__ == '__main__':
    unittest.main()
