"""volume.py - representation of a Terastitcher volume"""

import numpy as np
import os
import re
import tifffile
from xml.etree import ElementTree

def get_dim_tuple(element):
    """Given an element, extract the Terastitcher z, y, x dimensions

    :param element: an etree DOM element
    :returns: a tuple of the Z, Y and X extracted from the V, H and D attributes
    of the element
    """
    return tuple([float(element.attrib[_]) for _ in "DVH"])

class VExtentBase:
    """A volume extent (in voxels)"""

    def intersects(self, other):
        """Determine whether two extents intersect

        :param other: another VExtent
        :returns: True if the volumes intersect
        """
        return self.x0 < other.x1 and self.x1 >= other.x0 and \
               self.y0 < other.y1 and self.y1 >= other.y0 and \
               self.z0 < other.z1 and self.z1 >= other.z0

    def intersection(self, other):
        """Return the intersection volume of two extents"""
        return VExtent(max(self.x0, other.x0), min(self.x1, other.x1),
                       max(self.y0, other.y0), min(self.y1, other.y1),
                       max(self.z0, other.z0), min(self.z1, other.z1))

    def contains(self, other):
        """Return True if the other volume is wholly within this one

        :param other: a VExtentBase volume extent
        """
        return self.x0 <= other.x0 and self.x1 >= other.x1 and \
               self.y0 <= other.y0 and self.y1 >= other.y1 and \
               self.z0 <= other.z0 and self.z1 >= other.z1

    @property
    def shape(self):
        """The number of voxels in the volume in the z, y and x directions"""
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)


class VExtent(VExtentBase):
    """A volume extent"""

    def __init__(self, x0, x1, y0, y1, z0, z1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1


class TSVStack(VExtentBase):
    def __init__(self, element, root_dir):
        """Initialize a stack from a "Stack" element

        :param element: an ElementTree element (with the "Stack" tag)
        :param root_dir: the root directory of the directory hierarchy
        """
        self.root_dir = root_dir
        self.n_chans = int(element.attrib["N_CHANS"])
        self.bytes_per_chan = int(element.attrib["N_BYTESxCHAN"])
        self.row = int(element.attrib["ROW"])
        self.column = int(element.attrib["COL"])
        self.x0 = int(element.attrib["ABS_H"])
        self.y0 = int(element.attrib["ABS_V"])
        self.z0 = int(element.attrib["ABS_D"])
        self.dir_name = element.attrib["DIR_NAME"]
        z_ranges = element.attrib["Z_RANGES"]
        z0, z1 = map(int, z_ranges[1:-1].split(","))
        if z_ranges.startswith("["):
            self.z0slice = z0
        else:
            self.z0slice = z0+1
        if z_ranges.endswith(")"):
            self.z1slice = z1
        else:
            self.z1slice = z1+1
        self.img_regex = element.attrib["IMG_REGEX"]
        self.__paths = None
        self.__x1 = None
        self.__y1 = None
        self.__dtype = None
        self.z1 = self.z0 + self.z1slice - self.z0slice

    @property
    def paths(self):
        """The paths to the individual slices"""
        if self.__paths is None:
            directory = os.path.join(self.root_dir, self.dir_name)
            self.__paths = []
            for filename in os.listdir(directory):
                if filename.endswith(".db"):
                    # Windows thumbnail file
                    continue
                if self.img_regex != "":
                    if not re.match(self.img_regex, filename):
                        continue
                self.__paths.append(os.path.join(directory, filename))
        return self.__paths

    def __set_x1y1(self):
        if self.__x1 is None:
            img = tifffile.imread(self.paths[0])
            self.__dtype = img.dtype
            height, width = img.shape[-2:]
            self.__x1 = self.x0 + width
            self.__y1 = self.y0 + height

    @property
    def x1(self):
        """The block's end coordinate in the x direction"""
        self.__set_x1y1()
        return self.__x1

    @property
    def y1(self):
        """The block's end coordinate in the y direction"""
        self.__set_x1y1()
        return self.__y1

    @property
    def dtype(self):
        """The numpy dtype of the array data"""
        self.__set_x1y1()
        return self.__dtype

    def imread(self, volume, result = None):
        """Read the image data from a block

        :param volume: the volume to read, a VExtent
        :param result: if not None, read into this array
        :returns: the requested volume
        """
        assert self.contains(volume)
        if result is None:
            result = np.zeros(self.shape, self.dtype)
        for z in range(volume.z0, volume.z1):
            plane = tifffile.imread(self.paths[z - self.z0])
            result[z - self.z0] = \
                plane[volume.y0 - self.y0:volume.y1 - self.y0,
                      volume.x0 - self.x0:volume.x1 - self.x0]
        return result


def compute_cosine(volume, stack, ostack, img):
    """Given two overlapping stacks, compute the cosine blend between them

    :param volume: the volume being blended
    :param stack: the stack from which the data is being taken
    :param ostack: the stack that overlaps it
    :param img: reduce the intensity using the cosine blend on this image
    """
    #
    # Figure out which side is overlapping
    #
    if ostack.x0 < volume.x1 and ostack.x1 > volume.x0:
        x0blend = ostack.x0
        x1blend = stack.x1
        blending = np.cos(np.linspace(0, np.pi/2, x1blend-x0blend+2)[1:-1])
        blending = blending[:volume.x1 - x0blend].astype(np.float32)
        blending_slice = \
            img[:, :, x0blend-volume.x0:].astype(np.float32) *\
            blending[np.newaxis, np.newaxis, :]
        img[:, :, x0blend-volume.x0:] = blending_slice.astype(img.dtype)
    elif volume.x0 < ostack.x1 and volume.y1 > ostack.x0:
        x0blend = stack.x0
        x1blend = ostack.x1
        blending = np.cos(np.linspace(np.pi/2, 0, x1blend-x0blend+2)[1:-1])
        blending = blending[volume.x0 - x0blend:]
        blending_slice = \
        img[:, :, x1blend - volume.x0:].astype(np.float32) * \
            blending[np.newaxis, np.newaxis, :]
        img[:, :, x1blend - volume.x0:] = blending_slice.astype(img.dtype)
    if ostack.y0 < volume.y1 and ostack.y1 > volume.y0:
        y0blend = ostack.y0
        y1blend = stack.y1
        blending = np.cos(np.linspace(0, np.pi/2, y1blend-y0blend+2)[1:-1])
        blending = blending[:volume.y1 - y0blend].astype(np.float32)
        blending_slice = \
            img[:, y0blend-volume.y0:].astype(np.float32) *\
            blending[np.newaxis, :, np.newaxis]
        img[:, y0blend-volume.y0:] = blending_slice.astype(img.dtype)
    elif volume.y0 < ostack.y1 and volume.y1 > ostack.y0:
        y0blend = stack.y0
        y1blend = ostack.y1
        blending = np.cos(np.linspace(np.pi/2, 0, y1blend-y0blend+2)[1:-1])
        blending = blending[volume.y0 - y0blend:]
        blending_slice = \
        img[:, y1blend - volume.y0:].astype(np.float32) * \
            blending[np.newaxis, :, np.newaxis]
        img[:, y1blend - volume.y0:] = blending_slice.astype(img.dtype)
    if ostack.z0 < volume.z1 and ostack.z1 > volume.z0:
        z0blend = ostack.z0
        z1blend = stack.z1
        blending = np.cos(np.linspace(0, np.pi/2, z1blend-z0blend+2)[1:-1])
        blending = blending[:volume.z1 - z0blend].astype(np.float32)
        blending_slice = \
            img[z0blend-volume.z0:].astype(np.float32) *\
            blending[:, np.newaxis, np.newaxis]
        img[z0blend-volume.z0:] = blending_slice.astype(img.dtype)
    elif volume.z0 < ostack.z1 and volume.z1 > ostack.z0:
        z0blend = stack.z0
        z1blend = ostack.z1
        blending = np.cos(np.linspace(np.pi/2, 0, z1blend-z0blend+2)[1:-1])
        blending = blending[volume.z0 - z0blend:]
        blending_slice = \
        img[z1blend - volume.z0:].astype(np.float32) * \
            blending[:, np.newaxis, np.newaxis]
        img[z1blend - volume.z0:] = blending_slice.astype(img.dtype)


class TSVVolume:
    def __init__(self, tree):
        """Initialize from an xml.etree.ElementTree

        :param tree: a tree, e.g. as loaded from ElementTree.parse(xml_path)
        """
        root = tree.getroot()
        assert root.tag == "TeraStitcher"
        self.input_plugin = root.attrib["input_plugin"]
        self.volume_format = root.attrib["volume_format"]
        self.stacks_dir = root.find("stacks_dir").attrib["value"]
        self.voxel_dims = get_dim_tuple(root.find("voxel_dims"))
        self.origin = get_dim_tuple(root.find("origin"))
        md = root.find("mechanical_displacements")
        self.mechanical_displacement_x = int(md.attrib["H"])
        self.mechanical_displacement_y = int(md.attrib["V"])
        dims = root.find("dimensions")
        self.stack_rows = int(dims.attrib["stack_rows"])
        self.stack_columns = int(dims.attrib["stack_columns"])
        self.stack_slices = int(dims.attrib["stack_slices"])
        stacks = root.find("STACKS")
        self.stacks = []
        for child in stacks.getchildren():
            if child.tag == "Stack":
                self.stacks.append(TSVStack(child, self.stacks_dir))

    @staticmethod
    def load(path):
        """Load a volume from an XML file"""
        tree = ElementTree.parse(path)
        return TSVVolume(tree)

    def imread(self, volume, dtype):
        """Read the given volume

        :param volume: a VExtent delimiting the volume to read
        :param dtype: the numpy dtype of the array to be returned
        :returns: the array corresponding to the volume (with zeros for
        data outside of the array).
        """
        result = np.zeros(volume.shape, dtype)
        intersections = []
        for stack in self.stacks:
            if stack.intersects(volume):
                intersections.append((stack, stack.intersection(volume)))

        for stack, intersection in intersections:
            part = stack.imread(intersection).astype(dtype)
            #
            # Look for overlaps and perform a cosine blending
            #
            inter_intersections = filter(
                lambda ostack, ointersection:
                intersection.intersects(ointersection),
                intersections)
            if len(inter_intersections) > 0:
                multiplier = np.ones(part.shape, np.float32)
                for ostack, ointersection in inter_intersections:
                    compute_cosine(intersection, stack, ostack, part)
            result[intersection.z0 - volume.z0:intersection.z1 - volume.z0,
                   intersection.y0 - volume.y0:intersection.y1 - volume.y0,
                   intersection.x0 - volume.x0:intersection.x1 - volume.x0] +=\
                part
        return result