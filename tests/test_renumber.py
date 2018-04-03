import contextlib
import os
import unittest
import shutil
import subprocess
import tempfile

from tsv.renumber import main


class TestRenumber(unittest.TestCase):

    @contextlib.contextmanager
    def make_case(self, file_names):
        tempdir = tempfile.mkdtemp()
        for file_name in file_names:
            path = os.path.join(tempdir, file_name)
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(path, "w") as fd:
                fd.write(file_name)
        yield tempdir
        shutil.rmtree(tempdir)

    def check_case(self, expected, my_files, root):
        for file_name, expected_file in zip(my_files, expected):
            path = os.path.join(root, expected_file)
            self.assertTrue(os.path.exists(path))
            with open(path) as fd:
                self.assertEqual(file_name, fd.read())

    def test_renumber_default_digits(self):
        my_files = [
            "123/456/1.tiff",
            "123/456/2.tiff",
            "123/789/1.tiff"
        ]
        expected = [
            "123/456/000001.tiff",
            "123/456/000002.tiff",
            "123/789/000001.tiff"
        ]
        with self.make_case(my_files) as root:
            main([root])
            self.check_case(expected, my_files, root)

    def test_renumber_custom_digits(self):
        my_files = [
            "123/456/1.tiff",
            "123/456/2.tiff",
            "123/789/1.tiff"
        ]
        expected = [
            "123/456/01.tiff",
            "123/456/02.tiff",
            "123/789/01.tiff"
        ]
        with self.make_case(my_files) as root:
            main(["--n-digits=2", root])
            self.check_case(expected, my_files, root)

    def test_command_line(self):
        subprocess.check_call(["tsv-renumber", "--help"])


if __name__ == '__main__':
    unittest.main()
