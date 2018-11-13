import contextlib
import os
import shutil
import tempfile
import unittest
from tsv.renumber_directories import main


@contextlib.contextmanager
def make_hierarchy(xs, ys):
    """Make a directory hierarchy for a test case

    :param xs: A sequence of X offsets
    :param ys: A sequence of Y offsets
    """
    root = tempfile.mkdtemp()
    for x in xs:
        xpath = os.path.join(root, "%06d" % x)
        os.mkdir(xpath)
        for y in ys:
            ypath = os.path.join(xpath, "%06d_%06d" % (x, y))
            os.mkdir(ypath)
    yield root
    shutil.rmtree(root)


class TestRenumberDirectories(unittest.TestCase):
    def test_no_renumber(self):
        xs = [10, 20]
        ys = [30, 40]
        with make_hierarchy(xs, ys) as root:
            main(["--path", root])
            for x in xs:
                self.assertTrue(os.path.exists(os.path.join(root, "%06d" % x)))
                for y in ys:
                    self.assertTrue(os.path.exists(os.path.join(
                        root, "%06d" % x, "%06d_%06d" % (x, y))))

    def test_renumber_y(self):
        xs = [10, 20]
        ys = [-30, -40]
        ydests = [10, 0]

        with make_hierarchy(xs, ys) as root:
            main(["--path", root])
            for x in xs:
                self.assertTrue(os.path.exists(os.path.join(root, "%06d" % x)))
                for y in ydests:
                    self.assertTrue(os.path.exists(os.path.join(
                        root, "%06d" % x, "%06d_%06d" % (x, y))))

    def test_renumber_x(self):
        xs = [-20, 20]
        xdests = [0, 40]
        ys = [30, 40]

        with make_hierarchy(xs, ys) as root:
            main(["--path", root])
            for x in xdests:
                self.assertTrue(os.path.exists(os.path.join(root, "%06d" % x)))
                for y in ys:
                    self.assertTrue(os.path.exists(os.path.join(
                        root, "%06d" % x, "%06d_%06d" % (x, y))))

    def test_renumber_xy(self):
        xs = [-20, 20]
        xdests = [0, 40]
        ys = [-30, -40]
        ydests = [10, 0]

        with make_hierarchy(xs, ys) as root:
            main(["--path", root])
            for x in xdests:
                self.assertTrue(os.path.exists(os.path.join(root, "%06d" % x)))
                for y in ydests:
                    self.assertTrue(os.path.exists(os.path.join(
                        root, "%06d" % x, "%06d_%06d" % (x, y))))

    def test_extra_files(self):
        xs = [-20, 20]
        xdests = [0, 40]
        ys = [-30, -40]
        ydests = [10, 0]

        with make_hierarchy(xs, ys) as root:
            os.mkdir(os.path.join(root, "foo"))
            os.mkdir(os.path.join(root, "%06d" % -20, "bar"))
            main(["--path", root])
            for x in xdests:
                self.assertTrue(os.path.exists(os.path.join(root, "%06d" % x)))
                for y in ydests:
                    self.assertTrue(os.path.exists(os.path.join(
                        root, "%06d" % x, "%06d_%06d" % (x, y))))

if __name__ == '__main__':
    unittest.main()
