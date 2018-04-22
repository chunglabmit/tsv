import contextlib
import numpy as np
import os
import shutil
import tempfile
import tifffile
import unittest
from tsv import volume
from tsv.raw import raw_imsave

xml_path = os.path.join(os.path.dirname(__file__), "example.xml")

FN_PATTERN = "%08d.tif"


@contextlib.contextmanager
def make_case(constant_value = None, use_raw=False):
    """Make a xml file and stacks for testing

    :param constant_value: make all pixels this value
    :param use_raw: if True, make a test case using .raw files.
    """
    tempdir = tempfile.mkdtemp()
    try:
        input_plugin = "raw" if use_raw else "tiff2D"
        xml_path = os.path.join(tempdir, "example.xml")
        with open(xml_path, "w") as fd:
            fd.write(xml_template.format(tempdir=tempdir,
                                         input_plugin=input_plugin))
        r = np.random.RandomState(1234)
        subdirs = [["000000/000000_000000",
                    "000000/000000_001000"],
                   ["001000/001000_000000",
                    "001000/001000_001000"]]
        if constant_value is None:
            stacks = [[
                [r.randint(0, 65535, (1024, 1024), np.uint16) for _ in range(2)]
                for __ in range(2)] for ___ in range(2)]
        else:
            stacks = [[
                [np.ones((1024, 1024), np.uint16) * constant_value
                 for _ in range(2)]
                for __ in range(2)] for ___ in range(2)]
        for row, (sds, stks) in enumerate(zip(subdirs, stacks)):
            for col, (subdir, stack) in enumerate(zip(sds, stks)):
                abs_subdir = os.path.join(tempdir, subdir)
                os.makedirs(abs_subdir)
                for i, plane in enumerate(stack):
                    filename = os.path.join(abs_subdir, FN_PATTERN % (i + 1))
                    if input_plugin == "raw":
                        filename = filename[:-3] + "raw"
                        raw_imsave(filename, plane)
                    else:
                        tifffile.imsave(filename, plane)
        yield xml_path, stacks
    finally:
        shutil.rmtree(tempdir)


class TestVExtent(unittest.TestCase):
    def test_intersects(self):
        for (x0a, x1a, y0a, y1a, z0a, z1a), \
            (x0b, x1b, y0b, y1b, z0b, z1b), \
            intersects in ((( 0, 10, 10, 20, 30, 40),
                            (10, 20, 10, 20, 30, 40), False),
                           ((0, 10, 10, 20, 30, 40),
                            (0, 10, 20, 30, 30, 40), False),
                           ((0, 10, 10, 20, 30, 40),
                            (0, 10, 10, 20, 40, 50), False),
                           (( 0, 10, 10, 20, 30, 40),
                            ( 9, 19, 10, 20, 30, 40), True),
                           ((0, 10, 10, 20, 30, 40),
                            (0, 20, 19, 29, 30, 40), True),
                           ((0, 10, 10, 20, 30, 40),
                            (0, 10, 10, 20, 39, 49), True)):
            v0 = volume.VExtent(x0a, x1a, y0a, y1a, z0a, z1a)
            v1 = volume.VExtent(x0b, x1b, y0b, y1b, z0b, z1b)
            self.assertEqual(v0.intersects(v1), intersects)
            self.assertEqual(v1.intersects(v0), intersects)

    def test_intersection(self):
        for (x0a, x1a, y0a, y1a, z0a, z1a), \
            (x0b, x1b, y0b, y1b, z0b, z1b), \
            (x0e, x1e, y0e, y1e, z0e, z1e), \
            in ((( 0, 10, 10, 20, 30, 40),
                 ( 5, 8, 10, 20, 30, 40),
                 ( 5, 8, 10, 20, 30, 40)),
                ((0, 10, 10, 20, 30, 40),
                 (0, 10, 15, 18, 30, 40),
                 (0, 10, 15, 18, 30, 40)),
                ((0, 10, 10, 20, 30, 40),
                 (0, 10, 10, 20, 35, 38),
                 (0, 10, 10, 20, 35, 38)),
                ((0, 10, 10, 20, 30, 40),
                 (9, 19, 10, 20, 30, 40),
                 (9, 10, 10, 20, 30, 40))):
            v0 = volume.VExtent(x0a, x1a, y0a, y1a, z0a, z1a)
            v1 = volume.VExtent(x0b, x1b, y0b, y1b, z0b, z1b)
            ve = volume.VExtent(x0e, x1e, y0e, y1e, z0e, z1e)
            for va, vb in ((v0, v1), (v1, v0)):
                vi = va.intersection(vb)
                self.assertEqual(vi.x0, ve.x0)
                self.assertEqual(vi.x1, ve.x1)
                self.assertEqual(vi.y0, ve.y0)
                self.assertEqual(vi.y1, ve.y1)
                self.assertEqual(vi.z0, ve.z0)
                self.assertEqual(vi.z1, ve.z1)

    def test_contains(self):
        for (x0a, x1a, y0a, y1a, z0a, z1a), \
            (x0b, x1b, y0b, y1b, z0b, z1b), \
            contains in ((( 0, 10, 10, 20, 30, 40),
                            (10, 20, 10, 20, 30, 40), False),
                           ((0, 10, 10, 20, 30, 40),
                            (0, 10, 20, 30, 30, 40), False),
                           ((0, 10, 10, 20, 30, 40),
                            (0, 10, 10, 20, 40, 50), False),
                           (( 0, 10, 10, 20, 30, 40),
                            ( 9, 19, 10, 20, 30, 40), False),
                           ((0, 10, 10, 20, 30, 40),
                            (0, 20, 19, 29, 30, 40), False),
                           ((0, 10, 10, 20, 30, 40),
                            (0, 10, 10, 20, 39, 49), False),
                           ((0, 10, 10, 20, 30, 40),
                            (1, 9, 11, 19, 31, 39), True)):
            v0 = volume.VExtent(x0a, x1a, y0a, y1a, z0a, z1a)
            v1 = volume.VExtent(x0b, x1b, y0b, y1b, z0b, z1b)
            self.assertEqual(v0.contains(v1), contains)


class TestTSVStack(unittest.TestCase):

    def test_xml(self):
        v = volume.TSVVolume.load(xml_path)
        s1 = v.stacks[0][1]
        self.assertIsInstance(s1, volume.TSVStack)
        self.assertEqual(s1.img_regex, "")
        self.assertEqual(s1.root_dir, v.stacks_dir)
        self.assertEqual(s1.bytes_per_chan, 2)
        self.assertEqual(s1.column, 1)
        self.assertEqual(s1.dir_name, "139100/139100_171560")
        self.assertEqual(s1.n_chans, 1)
        self.assertEqual(s1.row, 0)
        self.assertEqual(s1.x0, 1703)
        self.assertEqual(s1.y0, 0)
        # s3 is -39 in z relative to s0 and s1 is -19 relative to s0
        self.assertEqual(s1.z0, 20)

    def test_paths(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            s0 = v.stacks[0][0]
            for i, path in enumerate(s0.paths):
                filename = FN_PATTERN % (i + 1)
                expected = os.path.join(v.stacks_dir, s0.dir_name, filename)
                self.assertEqual(path, expected)

    def test_x1_y1_dtype(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            for s in sum(v.stacks, []):
                self.assertEqual(s.x1, s.x0 + 1024)
                self.assertEqual(s.y1, s.y0 + 1024)
                self.assertEqual(s.dtype, stacks[0][0][0].dtype)

    def test_imread(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(0, 10, 11, 21, 0, 2)
            img = v.stacks[0][0].imread(e)
            np.testing.assert_equal(img[0], stacks[0][0][0][11:21, :10])
            np.testing.assert_equal(img[1], stacks[0][0][1][11:21, :10])

    def test_imread_raw(self):
        with make_case(use_raw=True) as ps:
            my_xml_path, stacks = ps
            v = volume.TSVVolume.load(my_xml_path)
            e = volume.VExtent(0, 10, 11, 21, 0, 2)
            img = v.stacks[0][0].imread(e)
            np.testing.assert_equal(img[0], stacks[0][0][0][11:21, :10])
            np.testing.assert_equal(img[1], stacks[0][0][1][11:21, :10])

    def test_imread_inplace(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(0, 10, 11, 21, 0, 2)
            img = np.zeros(e.shape)
            v.stacks[0][0].imread(e, img)
            np.testing.assert_equal(img[0], stacks[0][0][0][11:21, :10])
            np.testing.assert_equal(img[1], stacks[0][0][1][11:21, :10])

    def test_read_z_offset(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(0, 10, 11, 21, 1, 2)
            img = np.zeros(e.shape)
            v.stacks[0][0].imread(e, img)
            np.testing.assert_equal(img[0], stacks[0][0][1][11:21, :10])


class TestComputeCosine(unittest.TestCase):
    def test_x(self):
        #
        # We test at the border, 1/2 is not overlapping and 1/2 is
        #
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(976, 1024, 0, 10, 0, 2)
            img = np.random.RandomState(1234).randint(0, 65535, (2, 10, 48))
            img = img.astype(np.float32)
            cosine = np.cos(np.arctan2(
                np.arange(1, 25).astype(float),
                np.arange(24, 0, -1).astype(float)))
            mult = np.hstack([np.ones(24), cosine ** 2])
            expected = img.astype(float) * mult[np.newaxis, np.newaxis, :]
            s0 = v.stacks[0][0]
            s1 = v.stacks[0][1]
            # Quick check to make sure we got the right stack
            self.assertEqual(s0.x0, 0)
            self.assertEqual(s0.y0, 0)
            self.assertEqual(s0.z0, 0)
            self.assertEqual(s1.x0, 1000)
            self.assertEqual(s1.y0, 0)
            self.assertEqual(s1.z0, 0)
            volume.compute_cosine(e, s0, s1, img)
            np.testing.assert_almost_equal(img, expected, 0)

    def test_inv_x(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(1000, 1048, 0, 10, 0, 2)
            img = np.random.RandomState(1234).randint(0, 65535, (2, 10, 48))\
                  .astype(np.float32)
            cosine = np.cos(np.arctan2(
                np.arange(24, 0, -1).astype(float),
                np.arange(1, 25).astype(float)))
            mult = np.hstack([cosine **2, np.ones(24)])
            expected = img * mult[np.newaxis, np.newaxis, :]
            s0 = v.stacks[0][0]
            s1 = v.stacks[0][1]
            # Quick check to make sure we got the right stack
            self.assertEqual(s0.x0, 0)
            self.assertEqual(s0.y0, 0)
            self.assertEqual(s0.z0, 0)
            self.assertEqual(s1.x0, 1000)
            self.assertEqual(s1.y0, 0)
            self.assertEqual(s1.z0, 0)
            volume.compute_cosine(e, s1, s0, img)
            np.testing.assert_almost_equal(img, expected, 0)

    def test_y(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(0, 10, 976, 1024, 0, 2)
            img = np.random.RandomState(1234).randint(0, 65535, (2, 48, 10))\
                  .astype(np.float32)
            cosine = np.cos(np.arctan2(
                np.arange(1, 25).astype(float),
                np.arange(24, 0, -1).astype(float)))
            mult = np.hstack([np.ones(24), cosine ** 2])
            expected = img.astype(float) * mult[np.newaxis, :, np.newaxis]
            s0 = v.stacks[0][0]
            s1 = v.stacks[1][0]
            # Quick check to make sure we got the right stack
            self.assertEqual(s0.x0, 0)
            self.assertEqual(s0.y0, 0)
            self.assertEqual(s0.z0, 0)
            self.assertEqual(s1.x0, 0)
            self.assertEqual(s1.y0, 1000)
            self.assertEqual(s1.z0, 0)
            volume.compute_cosine(e, s0, s1, img)
            np.testing.assert_almost_equal(img, expected.astype(np.uint16), 0)

    def test_inv_y(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(0, 10, 1000, 1048, 0, 2)
            img = np.random.RandomState(1234).randint(0, 65535, (2, 48, 10))\
                  .astype(np.float32)
            cosine = np.cos(np.arctan2(
                np.arange(24, 0, -1).astype(float),
                np.arange(1, 25).astype(float)))
            mult = np.hstack((cosine ** 2, np.ones(24)))
            expected = img.astype(float) * mult[np.newaxis, :, np.newaxis]
            s0 = v.stacks[0][0]
            s1 = v.stacks[1][0]
            # Quick check to make sure we got the right stack
            self.assertEqual(s0.x0, 0)
            self.assertEqual(s0.y0, 0)
            self.assertEqual(s0.z0, 0)
            self.assertEqual(s1.x0, 0)
            self.assertEqual(s1.y0, 1000)
            self.assertEqual(s1.z0, 0)
            volume.compute_cosine(e, s1, s0, img)
            np.testing.assert_almost_equal(img, expected.astype(np.uint16), 0)


class TestGetDistanceFromEdge(unittest.TestCase):
    def test_xmin(self):
        #
        # Test where the edge is the x minimum
        #
        stack = volume.VExtent(10, 30, 10, 35, 20, 50)
        ostack = volume.VExtent(0, 20, 10, 35, 20, 50)
        v = stack.intersection(ostack)
        d = volume.get_distance_from_edge(v, stack, ostack)
        expected = np.array([[np.arange(1, 11)] * 25] * 30)
        np.testing.assert_equal(d, expected)

    def test_xmax(self):
        #
        # Test where the edge is the x minimum
        #
        stack = volume.VExtent(0, 20, 10, 35, 20, 50)
        ostack = volume.VExtent(10, 30, 10, 35, 20, 50)
        v = stack.intersection(ostack)
        d = volume.get_distance_from_edge(v, stack, ostack)
        expected = np.array([[np.arange(10, 0, -1)] * 25] * 30)
        np.testing.assert_equal(d, expected)

    def test_ymin(self):
        #
        # Test where the edge is the x minimum
        #
        stack = volume.VExtent(10, 30, 20, 35, 20, 50)
        ostack = volume.VExtent(10, 30, 10, 35, 20, 50)
        v = stack.intersection(ostack)
        d = volume.get_distance_from_edge(v, stack, ostack)
        expected = np.array([[[i] * 20 for i in range(1, 16)]] * 30)
        np.testing.assert_equal(d, expected)

    def test_ymax(self):
        #
        # Test where the edge is the x minimum
        #
        stack = volume.VExtent(10, 30, 10, 25, 20, 50)
        ostack = volume.VExtent(10, 30, 10, 35, 20, 50)
        v = stack.intersection(ostack)
        d = volume.get_distance_from_edge(v, stack, ostack)
        expected = np.array([[[i] * 20 for i in range(15, 0, -1)]] * 30)
        np.testing.assert_equal(d, expected)

    def test_zmin(self):
        #
        # Test where the edge is the x minimum
        #
        stack = volume.VExtent(10, 30, 10, 35, 30, 50)
        ostack = volume.VExtent(10, 30, 10, 35, 20, 50)
        v = stack.intersection(ostack)
        d = volume.get_distance_from_edge(v, stack, ostack)
        expected = np.array([[[i] * 20] * 25 for i in range(1, 21)])
        np.testing.assert_equal(d, expected)

    def test_zmax(self):
        #
        # Test where the edge is the x minimum
        #
        stack = volume.VExtent(10, 30, 10, 35, 20, 45)
        ostack = volume.VExtent(10, 30, 10, 35, 20, 50)
        v = stack.intersection(ostack)
        d = volume.get_distance_from_edge(v, stack, ostack)
        expected = np.array([[[i] * 20] * 25 for i in range(25, 0, -1)])
        np.testing.assert_equal(d, expected)

    def test_two(self):
        #
        # Test two at once
        #
        stack = volume.VExtent(10, 30, 20, 35, 20, 45)
        ostack = volume.VExtent(10, 30, 10, 35, 20, 50)
        v = stack.intersection(ostack)
        d = volume.get_distance_from_edge(v, stack, ostack)
        expected_y = np.array([[[i] * 20 for i in range(1, 16)]] * 25)
        expected_z = np.array([[[i] * 20] * 15 for i in range(25, 0, -1)])
        expected = np.minimum(expected_y, expected_z)
        np.testing.assert_equal(d, expected)


class TestTSVVolume(unittest.TestCase):

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
        self.assertEqual(len(v.stacks), 2)
        self.assertEqual(len(v.stacks[0]), 2)
        self.assertEqual(len(v.stacks[1]), 2)

    def test_flattened_stacks(self):
        v = volume.TSVVolume.load(xml_path)
        for fs, s in zip(v.flattened_stacks(),
                         (v.stacks[0][0], v.stacks[0][1],
                          v.stacks[1][0], v.stacks[1][1])):
            self.assertEqual(fs, s)

    def test_imread_no_overlap(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(0, 10, 20, 30, 0, 2)
            img = v.imread(e, np.uint16)
            for z in range(2):
                np.testing.assert_equal(img[z], stacks[0][0][z][20:30, :10])

    def test_imread_overlap(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(976, 1048, 976, 1048, 0, 2)
            expected = np.zeros(e.shape, np.float32)
            v00 = volume.VExtent(976, 1024, 976, 1024, 0, 2)
            s00 = v.stacks[0][0]
            i00 = s00.imread(v00).astype(np.float32)
            v01 = volume.VExtent(1000, 1048, 976, 1024, 0, 2)
            s01 = v.stacks[0][1]
            i01 = s01.imread(v01).astype(np.float32)
            v10 = volume.VExtent(976, 1024, 1000, 1048, 0, 2)
            s10 = v.stacks[1][0]
            i10 = s10.imread(v10).astype(np.float32)
            v11 = volume.VExtent(1000, 1048, 1000, 1048, 0, 2)
            s11 = v.stacks[1][1]
            i11 = s11.imread(v11).astype(np.float32)
            m00, m01, m10, m11 = np.ones((4, 2, 48, 48), np.float32)
            m = np.zeros((2, 72, 72), np.float32)
            volume.compute_cosine(v00, s00, s01, i00)
            volume.compute_cosine(v00, s00, s10, i00)
            volume.compute_cosine(v00, s00, s11, i00)
            volume.compute_cosine(v00, s00, s01, m00)
            volume.compute_cosine(v00, s00, s10, m00)
            volume.compute_cosine(v00, s00, s11, m00)
            expected[:, :48, :48] += i00.astype(expected.dtype)
            m[:, :48, :48] += m00
            volume.compute_cosine(v01, s01, s00, i01)
            volume.compute_cosine(v01, s01, s10, i01)
            volume.compute_cosine(v01, s01, s11, i01)
            volume.compute_cosine(v01, s01, s00, m01)
            volume.compute_cosine(v01, s01, s10, m01)
            volume.compute_cosine(v01, s01, s11, m01)
            expected[:, :48, -48:] += i01.astype(expected.dtype)
            m[:, :48, -48:] += m01
            volume.compute_cosine(v10, s10, s00, i10)
            volume.compute_cosine(v10, s10, s01, i10)
            volume.compute_cosine(v10, s10, s11, i10)
            volume.compute_cosine(v10, s10, s00, m10)
            volume.compute_cosine(v10, s10, s01, m10)
            volume.compute_cosine(v10, s10, s11, m10)
            expected[:, -48:, :48] += i10.astype(expected.dtype)
            m[:, -48:, :48] += m10
            volume.compute_cosine(v11, s11, s00, i11)
            volume.compute_cosine(v11, s11, s01, i11)
            volume.compute_cosine(v11, s11, s10, i11)
            volume.compute_cosine(v11, s11, s00, m11)
            volume.compute_cosine(v11, s11, s01, m11)
            volume.compute_cosine(v11, s11, s10, m11)
            expected[:, -48:, -48:] += i11.astype(expected.dtype)
            m[:, -48:, -48:] += m11
            expected = np.around(expected / m)
            img = v.imread(e, np.uint16)
            np.testing.assert_almost_equal(img, expected, 0)

    def test_constant_value(self):
        with make_case(constant_value=100) as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            img = v.imread(v.volume, np.uint16)
            self.assertTrue(np.all(np.abs(img.astype(np.int32) - 100) < 2))

    def test_diagnostic_img(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = volume.VExtent(1000, 1024, 1000, 1024, 0, 2)
            img = v.make_diagnostic_img(e)
            self.assertTupleEqual(tuple(img.shape[:-1]), tuple(e.shape))
            self.assertEqual(img.shape[-1], 4)
            for z in range(2):
                np.testing.assert_equal(img[z, :, :, 0],
                                        stacks[0][0][z][-24:, -24:])
                np.testing.assert_equal(img[z, :, :, 1],
                                        stacks[0][1][z][-24:, :24])
                np.testing.assert_equal(img[z, :, :, 2],
                                        stacks[1][0][z][:24, -24:])
                np.testing.assert_equal(img[z, :, :, 3],
                                        stacks[1][1][z][:24, :24])


    def test_volume(self):
        with make_case() as ps:
            xml_path, stacks = ps
            v = volume.TSVVolume.load(xml_path)
            e = v.volume
            self.assertEqual(e.x0, 0)
            self.assertEqual(e.x1, 2024)
            self.assertEqual(e.y0, 0, 2024)
            self.assertEqual(e.z0, 0)
            self.assertEqual(e.z1, 2)

"""The XML template consists of four stacks which overlap by 24 voxels
"""
xml_template = """<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE TeraStitcher SYSTEM "TeraStitcher.DTD">
<TeraStitcher volume_format="TiledXY|2Dseries" input_plugin="{input_plugin:s}">
    <stacks_dir value="{tempdir:s}" />
    <voxel_dims V="1.599" H="1.599" D="1" />
    <origin V="17.156" H="11.126" D="10" />
    <mechanical_displacements V="2783" H="2784" />
    <dimensions stack_rows="2" stack_columns="2" stack_slices="2" />
    <STACKS>
        <Stack N_CHANS="1" N_BYTESxCHAN="2" ROW="0" COL="0" ABS_V="0" ABS_H="0" 
               ABS_D="0" STITCHABLE="no" DIR_NAME="000000/000000_000000" 
               Z_RANGES="[0,2)" IMG_REGEX="">
            <NORTH_displacements />
            <EAST_displacements>
                <Displacement TYPE="MIP_NCC">
                    <V displ="0" default_displ="0" reliability="0.787502"
                     nccPeak="0.840414" nccWidth="7" nccWRangeThr="25" 
                     nccInvWidth="26" delay="25" />
                    <H displ="1000" default_displ="1741" reliability="0.829253" 
                       nccPeak="0.885216" nccWidth="6" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                    <D displ="0" default_displ="0" reliability="0.722427" 
                       nccPeak="0.843184" nccWidth="11" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                </Displacement>
            </EAST_displacements>
            <SOUTH_displacements>
                <Displacement TYPE="MIP_NCC">
                    <V displ="1000" default_displ="1741" reliability="0.875742"
                       nccPeak="0.866778" nccWidth="3" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                    <H displ="0" default_displ="0" reliability="0.879403"
                       nccPeak="0.874159" nccWidth="3" nccWRangeThr="25"
                        nccInvWidth="26" delay="25" />
                    <D displ="0" default_displ="0" reliability="0.841427"
                       nccPeak="0.9079" nccWidth="6" nccWRangeThr="25" \
                       nccInvWidth="26" delay="25" />
                </Displacement>
            </SOUTH_displacements>
            <WEST_displacements />
        </Stack>
        <Stack N_CHANS="1" N_BYTESxCHAN="2" ROW="0" COL="1" ABS_V="0" 
               ABS_H="1000" ABS_D="0" STITCHABLE="no" 
               DIR_NAME="000000/000000_001000" Z_RANGES="[0,2)" IMG_REGEX="">
            <NORTH_displacements />
            <EAST_displacements/>
            <SOUTH_displacements>
                <Displacement TYPE="MIP_NCC">
                    <V displ="1000" default_displ="1741" reliability="0.875742"
                       nccPeak="0.866778" nccWidth="3" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                    <H displ="0" default_displ="0" reliability="0.879403"
                       nccPeak="0.874159" nccWidth="3" nccWRangeThr="25"
                        nccInvWidth="26" delay="25" />
                    <D displ="0" default_displ="0" reliability="0.841427"
                       nccPeak="0.9079" nccWidth="6" nccWRangeThr="25" \
                       nccInvWidth="26" delay="25" />
                </Displacement>
            </SOUTH_displacements>
            <WEST_displacements>
                <Displacement TYPE="MIP_NCC">
                    <V displ="0" default_displ="0" reliability="0.787502"
                     nccPeak="0.840414" nccWidth="7" nccWRangeThr="25" 
                     nccInvWidth="26" delay="25" />
                    <H displ="-1000" default_displ="1741" reliability="0.829253" 
                       nccPeak="0.885216" nccWidth="6" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                    <D displ="0" default_displ="0" reliability="0.722427" 
                       nccPeak="0.843184" nccWidth="11" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                </Displacement>
            </WEST_displacements>
        </Stack>
        <Stack N_CHANS="1" N_BYTESxCHAN="2" ROW="1" COL="0" ABS_V="1000" 
               ABS_H="0" ABS_D="0" STITCHABLE="no" 
               DIR_NAME="001000/001000_000000" Z_RANGES="[0,2)" IMG_REGEX="">
            <NORTH_displacements>
                <Displacement TYPE="MIP_NCC">
                    <V displ="-1000" default_displ="1741" reliability="0.875742"
                       nccPeak="0.866778" nccWidth="3" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                    <H displ="0" default_displ="0" reliability="0.879403"
                       nccPeak="0.874159" nccWidth="3" nccWRangeThr="25"
                        nccInvWidth="26" delay="25" />
                    <D displ="0" default_displ="0" reliability="0.841427"
                       nccPeak="0.9079" nccWidth="6" nccWRangeThr="25" \
                       nccInvWidth="26" delay="25" />
                </Displacement>
            </NORTH_displacements>
            <EAST_displacements>
                <Displacement TYPE="MIP_NCC">
                    <V displ="0" default_displ="0" reliability="0.787502"
                     nccPeak="0.840414" nccWidth="7" nccWRangeThr="25" 
                     nccInvWidth="26" delay="25" />
                    <H displ="1000" default_displ="1741" reliability="0.829253" 
                       nccPeak="0.885216" nccWidth="6" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                    <D displ="0" default_displ="0" reliability="0.722427" 
                       nccPeak="0.843184" nccWidth="11" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                </Displacement>
            </EAST_displacements>
            <SOUTH_displacements/>
            <WEST_displacements/>
        </Stack>
        <Stack N_CHANS="1" N_BYTESxCHAN="2" ROW="1" COL="1" ABS_V="1000"
               ABS_H="1000" ABS_D="0" STITCHABLE="no"
               DIR_NAME="001000/001000_001000" Z_RANGES="[0,2)" IMG_REGEX="">
            <NORTH_displacements>
                <Displacement TYPE="MIP_NCC">
                    <V displ="-1000" default_displ="1741" reliability="0.875742"
                       nccPeak="0.866778" nccWidth="3" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                    <H displ="0" default_displ="0" reliability="0.879403"
                       nccPeak="0.874159" nccWidth="3" nccWRangeThr="25"
                        nccInvWidth="26" delay="25" />
                    <D displ="0" default_displ="0" reliability="0.841427"
                       nccPeak="0.9079" nccWidth="6" nccWRangeThr="25" \
                       nccInvWidth="26" delay="25" />
                </Displacement>
            </NORTH_displacements>
            <EAST_displacements/>
            <SOUTH_displacements/>
            <WEST_displacements>
                <Displacement TYPE="MIP_NCC">
                    <V displ="0" default_displ="0" reliability="0.787502"
                     nccPeak="0.840414" nccWidth="7" nccWRangeThr="25" 
                     nccInvWidth="26" delay="25" />
                    <H displ="-1000" default_displ="1741" reliability="0.829253" 
                       nccPeak="0.885216" nccWidth="6" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                    <D displ="0" default_displ="0" reliability="0.722427" 
                       nccPeak="0.843184" nccWidth="11" nccWRangeThr="25" 
                       nccInvWidth="26" delay="25" />
                </Displacement>
            </WEST_displacements>
        </Stack>
    </STACKS>
</TeraStitcher>
"""
