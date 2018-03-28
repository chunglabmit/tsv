"""volume.py - representation of a Terastitcher volume"""

import abc
import enum
import itertools
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


class Location:
    """A coordinate location"""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, item):
        """Get a coordinate by index"""
        assert 0 <= item <= 2
        if item == 0:
            return self.z
        if item == 1:
            return self.y
        return self.x

    def __repr__(self):
        return "{{x={x:d}, y={y:d}, z={z:d}}}".format(
            x=self.x, y=self.y, z=self.z)


class VExtentBase(abc.ABC):
    """A volume extent (in voxels)"""

    def intersects(self, other):
        """Determine whether two extents intersect

        :param other: another VExtent
        :returns: True if the volumes intersect
        """
        return self.x0 < other.x1 and self.x1 > other.x0 and \
               self.y0 < other.y1 and self.y1 > other.y0 and \
               self.z0 < other.z1 and self.z1 > other.z0

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
    @abc.abstractmethod
    def x0(self):
        pass

    @property
    @abc.abstractmethod
    def x1(self):
        pass

    @property
    @abc.abstractmethod
    def y0(self):
        pass

    @property
    @abc.abstractmethod
    def y1(self):
        pass

    @property
    @abc.abstractmethod
    def z0(self):
        pass

    @property
    @abc.abstractmethod
    def z1(self):
        pass

    @property
    def shape(self):
        """The number of voxels in the volume in the z, y and x directions"""
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    def start(self, idx):
        """The start coordinate at the given index

        :param idx: an index between 0 and 2 - 0 = z, 1 = y, 2 = x
        :returns: the x0, y0 or z0 of the volume
        """
        assert 0 <= idx <= 2
        if idx == 0:
            return self.z0
        elif idx == 1:
            return self.y0
        return self.x0

    def end(self, idx):
        """The end coordinate at the given index

        :param idx: an index between 0 and 2 - 0 = z, 1 = y, 2 = x
        :returns: the x1, y1 or z1 of the volume
        """
        assert 0 <= idx <= 2
        if idx == 0:
            return self.z1
        elif idx == 1:
            return self.y1
        return self.x1

    def __repr__(self):
        return "VExtent(x0={x0}, x1={x1}, y0={y0}, y1={y1}, z0={z0}, z1={z1})"\
               .format(x0=self.x0, y0=self.y0, z0=self.z0,
                       x1=self.x1, y1=self.y1, z1=self.z1)


class VExtent(VExtentBase):
    """A volume extent"""

    def __init__(self, x0, x1, y0, y1, z0, z1):
        self._x0 = x0
        self._x1 = x1
        self._y0 = y0
        self._y1 = y1
        self._z0 = z0
        self._z1 = z1

    @property
    def x0(self):
        return self._x0

    @property
    def x1(self):
        return self._x1

    @property
    def y0(self):
        return self._y0

    @property
    def y1(self):
        return self._y1

    @property
    def z0(self):
        return self._z0

    @property
    def z1(self):
        return self._z1


class TSVStack(VExtentBase):
    def __init__(self, element, offset:Location, root_dir,
                 ordering_pattern="[^0-9]*(\\d+).*\\.tiff?"):
        """Initialize a stack from a "Stack" element

        :param element: an ElementTree element (with the "Stack" tag)
        :param root_dir: the root directory of the directory hierarchy
        :param ordering_pattern: how to find the image order # - an expression
        that extracts a numeric z from the path name.
        """
        self.root_dir = root_dir
        self.n_chans = int(element.attrib["N_CHANS"])
        self.bytes_per_chan = int(element.attrib["N_BYTESxCHAN"])
        self.row = int(element.attrib["ROW"])
        self.column = int(element.attrib["COL"])
        self._x0 = offset.x
        self._y0 = offset.y
        self._z0 = offset.z
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
        self.ordering_pattern = ordering_pattern
        self.__paths = None
        self.__x1 = None
        self.__y1 = None
        self.__dtype = None

    @property
    def paths(self):
        """The paths to the individual slices"""
        if self.__paths is None:
            directory = os.path.join(self.root_dir, self.dir_name)
            my_paths = []
            for filename in sorted(os.listdir(directory)):
                match = re.match(self.ordering_pattern, filename)
                if not match:
                    continue
                ordering = int(match.groups()[0])
                if self.img_regex != "":
                    if not re.match(self.img_regex, filename):
                        continue
                my_paths.append((ordering, os.path.join(directory, filename)))
            self.__paths = [_[1] for _ in sorted(my_paths)]
        return self.__paths

    def __set_x1y1(self):
        if self.__x1 is None:
            img = tifffile.imread(self.paths[0])
            self.__dtype = img.dtype
            height, width = img.shape[-2:]
            self.__x1 = self.x0 + width
            self.__y1 = self.y0 + height

    @property
    def x0(self):
        return self._x0

    @property
    def x1(self):
        """The block's end coordinate in the x direction"""
        self.__set_x1y1()
        return self.__x1

    @property
    def y0(self):
        return self._y0

    @property
    def y1(self):
        """The block's end coordinate in the y direction"""
        self.__set_x1y1()
        return self.__y1

    @property
    def z0(self):
        return self._z0

    @property
    def z1(self):
        return self._z0 + self.z1slice - self.z0slice

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
            result = np.zeros(volume.shape, self.dtype)
        for z in range(volume.z0, volume.z1):
            plane = tifffile.imread(self.paths[z - self.z0])
            result[z - volume.z0] = \
                plane[volume.y0 - self.y0:volume.y1 - self.y0,
                      volume.x0 - self.x0:volume.x1 - self.x0]
        return result


def compute_cosine(volume:VExtentBase, stack:TSVStack, ostack:TSVStack, img):
    """Given two overlapping stacks, compute the cosine blend between them

    :param volume: the volume being blended
    :param stack: the stack from which the data is being taken
    :param ostack: the stack that overlaps it
    :param img: reduce the intensity using the cosine blend on this image
    """
    if not volume.intersects(ostack):
        return
    iv = volume.intersection(ostack)
    #
    # Find the distance to the edge of the intersecting volume per voxel
    #
    d = get_distance_from_edge(iv, stack, ostack)
    od = get_distance_from_edge(iv, ostack, stack)
    if np.min(d) == np.inf:
        d[:] = np.max(od)
    elif np.min(od) == np.inf:
        od[:] = np.max(d)
    #
    # Use the ratio of the two distances to get an "angle". The angle will
    # be 45 degrees if the voxels are the same distance away from the edge
    # and the two stacks will be equally blended. If there is a big difference
    # then the blending will favor the volume that is further away.
    #
    angle = np.arctan2(d, od)
    blending = np.sin(angle) ** 2
    img[
        iv.z0 - volume.z0:iv.z1 - volume.z0,
        iv.y0 - volume.y0:iv.y1 - volume.y0,
        iv.x0 - volume.x0:iv.x1 - volume.x0] *= blending.astype(img.dtype)


class Edge(enum.Flag):
    """Keep track of which edge or edges have some property"""

    """The X0 edge of the volume"""
    XMIN=enum.auto()

    """The X1 edge of the volume"""
    XMAX=enum.auto()

    """The Y0 edge of the volume"""
    YMIN = enum.auto()

    """The Y1 edge of the volume"""
    YMAX = enum.auto()

    """The Z0 edge of the volume"""
    ZMIN = enum.auto()

    """The Z1 edge of the volume"""
    ZMAX = enum.auto()


def get_distance_from_edge(tgt:VExtentBase, stack:VExtentBase, ostack:VExtentBase) -> np.ndarray:
    """For the volume, get the distance per voxel to the nearest edge

    :param tgt: the target volume to be filled
    :param stack: The stack on which to make the distance estimate
    :param ostack: The stack that is overlapping
    :returns: an array, similarly sized to the overlap volume, giving the
              minimum distance to the nearest edge.
    """
    edges = Edge(0)
    if ostack.x1 > stack.x0 > ostack.x0:
        edges = edges | Edge.XMIN
    if ostack.x0 < stack.x1 < ostack.x1:
        edges = edges | Edge.XMAX
    if ostack.y1 > stack.y0 > ostack.y0:
        edges = edges | Edge.YMIN
    if ostack.y0 < stack.y1 < ostack.y1:
        edges = edges | Edge.YMAX
    if ostack.z1 > stack.z0 > ostack.z0:
        edges = edges | Edge.ZMIN
    if ostack.z0 < stack.z1 < ostack.z1:
        edges = edges | Edge.ZMAX
    volume = stack.intersection(ostack)
    assert volume.contains(tgt)
    #
    # Start out with all voxels maximally far from the edge
    #
    max_distance = np.inf
    if ostack.x1 != stack.x1 and ostack.x0 != stack.x0:
        max_distance = volume.shape[2]
    if ostack.y1 != stack.y1 and ostack.y0 != stack.y0:
        max_distance = min(max_distance, volume.shape[1])
    if ostack.z1 != stack.z1 and ostack.z0 != stack.z0:
        max_distance = min(max_distance, volume.shape[0])
    result = np.ones(tgt.shape, np.float32) * max_distance
    #
    # Process the starting edges
    #
    for idx, flag in enumerate((Edge.ZMIN, Edge.YMIN, Edge.XMIN)):
        if edges & flag:
            slices = [np.newaxis] * 3
            slices[idx] = slice(0, tgt.shape[idx])
            result = np.minimum(
                result,
                np.arange(tgt.start(idx) - volume.start(idx) + 1,
                          tgt.end(idx) - volume.start(idx) + 1)[tuple(slices)])
    #
    # Process the ending edges
    #
    for idx, flag in enumerate((Edge.ZMAX, Edge.YMAX, Edge.XMAX)):
        if edges & flag:
            slices = [np.newaxis] * 3
            slices[idx] = slice(0, tgt.shape[idx])
            result = np.minimum(
                result,
                np.arange(volume.end(idx) - tgt.start(idx),
                          volume.end(idx) - tgt.end(idx), -1)[tuple(slices)])
    return result


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
        self.make_stacks(root)

    def make_stacks(self, root):
        """Parse and properly offset the stacks

        :param root: the root node of the xml
        """
        stacks = root.find("STACKS")
        selems = [[ None] * self.stack_columns for _ in range(self.stack_rows)]
        self.stacks = [[ None] * self.stack_columns
                       for _ in range(self.stack_rows)]
        self.offsets = [[ None] * self.stack_columns
                        for _ in range(self.stack_rows)]
        self.offsets[0][0] = Location(0, 0, 0)
        for child in stacks.getchildren():
            if child.tag == "Stack":
                row = int(child.attrib["ROW"])
                column = int(child.attrib["COL"])
                selems[row][column] = child
        for row, elements in enumerate(selems):
            for column, child in enumerate(elements):
                if row > 0:
                    prev = self.offsets[row-1][column]
                    dn = child.find("NORTH_displacements").getchildren()[0]
                    xoff = -int(dn.find("H").attrib["displ"])
                    yoff = -int(dn.find("V").attrib["displ"])
                    zoff = -int(dn.find("D").attrib["displ"])
                    offset = Location(prev.x + xoff,
                                      prev.y + yoff,
                                      prev.z + zoff)
                    self.offsets[row][column] = offset
                elif column > 0:
                    prev = self.offsets[row][column-1]
                    dn = child.find("WEST_displacements").getchildren()[0]
                    xoff = -int(dn.find("H").attrib["displ"])
                    yoff = -int(dn.find("V").attrib["displ"])
                    zoff = -int(dn.find("D").attrib["displ"])
                    offset = Location(prev.x + xoff,
                                      prev.y + yoff,
                                      prev.z + zoff)
                    self.offsets[row][column] = offset
        #
        # Find the minimum absolute offset for x, y, z
        #
        min_x = min_y = min_z = np.iinfo(np.uint32).max
        for offset in sum(self.offsets, []):
            min_x = min(min_x, offset.x)
            min_y = min(min_y, offset.y)
            min_z = min(min_z, offset.z)
        #
        # Rebase the offsets so that coordinates are all positive and start
        # at zero
        #
        for row, column in itertools.product(range(self.stack_rows),
                                             range(self.stack_columns)):
            offset = self.offsets[row][column]
            offset = Location(offset.x - min_x,
                              offset.y - min_y,
                              offset.z - min_z)
            self.offsets[row][column] = offset
            self.stacks[row][column] = TSVStack(selems[row][column],
                                                offset,
                                                self.stacks_dir)


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
        result = np.zeros(volume.shape, np.float32)
        multiplier = np.zeros(volume.shape, np.float32)
        intersections = []
        for stack in sum(self.stacks, []):
            if stack.intersects(volume):
                intersections.append((stack, stack.intersection(volume)))

        for stack, intersection in intersections:
            part = stack.imread(intersection).astype(np.float32)
            mpart = np.ones(intersection.shape, np.float32)
            #
            # Look for overlaps and perform a cosine blending
            #
            inter_intersections = []
            for ostack, ointersection in intersections:
                if ostack == stack:
                    continue
                if ointersection.intersects(intersection):
                    inter_intersections.append((ostack, ointersection))
            if len(inter_intersections) > 0:
                for ostack, ointersection in inter_intersections:
                    compute_cosine(intersection, stack, ostack, part)
                    compute_cosine(intersection, stack, ostack, mpart)
            result[intersection.z0 - volume.z0:intersection.z1 - volume.z0,
                   intersection.y0 - volume.y0:intersection.y1 - volume.y0,
                   intersection.x0 - volume.x0:intersection.x1 - volume.x0] +=\
                part
            multiplier[
                intersection.z0 - volume.z0:intersection.z1 - volume.z0,
                intersection.y0 - volume.y0:intersection.y1 - volume.y0,
                intersection.x0 - volume.x0:intersection.x1 - volume.x0] += \
                mpart
        result = result / (multiplier + np.finfo(np.float32).eps)
        if np.dtype(dtype).kind in ("u", "i"):
            result = np.clip(np.around(result, 0),
                             np.iinfo(dtype).min,
                             np.iinfo(dtype).max)
        return result.astype(dtype)

    @property
    def volume(self) -> VExtent:
        """The VExtent of the volume"""
        x0 = y0 = z0 = np.iinfo(np.int32).max
        x1 = y1 = z1 = 0
        for stack in sum(self.stacks, []):
            x0 = min(x0, stack.x0)
            x1 = max(x1, stack.x1)
            y0 = min(y0, stack.y0)
            y1 = max(y1, stack.y1)
            z0 = min(z0, stack.z0)
            z1 = max(z1, stack.z1)
        return VExtent(x0, x1, y0, y1, z0, z1)

    @property
    def dtype(self):
        """The dtype inferred from the stack's bit-depth"""
        if self.stacks[0][0].bytes_per_chan == 1:
            return np.uint8
        elif self.stacks[0][0].bytes_per_chan == 2:
            return np.uint16
        else:
            return np.uint32
