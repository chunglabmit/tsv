"""raw.py - read and write .raw files


"""

import numpy as np

def raw_imread(path):
    """Read a .raw file

    :param path: path to the file
    :returns: a Numpy read-only array mapping the file as an image
    """
    as_uint32 = np.memmap(
        path,
        dtype=">u4",
        mode="r", shape=(2,))
    width, height = as_uint32[:2]
    return np.memmap(path,
                     dtype=">u2",
                     mode="r",
                     offset=8,
                     shape=(height, width))

def raw_imsave(path, img):
    """Write a .raw file

    :param path: path to the file
    :param img: a Numpy 2d array
    """

    as_uint32 = np.memmap(
        path,
        dtype=">u4",
        mode="w+", shape=(2,))
    as_uint32[0] = img.shape[1]
    as_uint32[1] = img.shape[0]
    del as_uint32
    as_uint16 = np.memmap(path,
                     dtype=">u2",
                     mode="r+",
                     offset=8,
                     shape=img.shape)
    as_uint16[:] = img
    del as_uint16