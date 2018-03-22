import numpy as np
import os
import unittest
from tsv import volume

xml_path = os.path.join(os.path.dirname(__file__), "example.xml")


class TestVolume(unittest.TestCase):
    def
    def test_xml(self):
        v = volume.TSVVolume.load(xml_path)
        self.assertEqual(v.input_plugin, "tiff2D")
        self.assertEqual(v.mechanical_displacement_x, 2784)
        self.assertEqual(v.mechanical_displacement_y, 2783)
        self.assertAlmostEqual(v.origin[0], 10)
        self.assertAlmostEqual(v.origin[1], 17.156)
        self.assertAlmostEqual(v.origin[2], 11.126)
        self.assertEqual(v.stack_columns, 2)
        self.assertEqual(v.stack_rows, 2)
        self.assertEqual(v.stack_slices, 4000)
        self.assertEqual(v.stacks_dir, "/home/user/data/stitching/Color_1")
        self.assertEqual(v.volume_format, "TiledXY|2Dseries")
        self.assertAlmostEqual(v.voxel_dims[0], 1)
        self.assertAlmostEqual(v.voxel_dims[1], 1.599)
        self.assertAlmostEqual(v.voxel_dims[2], 1.599)
        self.assertEqual(len(v.stacks), 4)
        s1 = v.stacks[1]
        self.assertIsInstance(s1, volume.TSVStack)
        self.assertEqual(s1.img_regex, "")
        self.assertEqual(s1.root_dir, v.stacks_dir)
        self.assertEqual(s1.bytes_per_chan, 2)
        self.assertEqual(s1.column, 1)
        self.assertEqual(s1.dir_name, "139100/139100_171560")
        self.assertEqual(s1.n_chans, 1)
        self.assertEqual(s1.row, 0)
        self.assertEqual(s1.x0, 1741)
        self.assertEqual(s1.y0, 0)
        self.assertEqual(s1.z0, 0)
