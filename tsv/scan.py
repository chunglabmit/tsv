import heapq
import itertools
import logging
import multiprocessing
import networkx as nx
import numpy as np
import os
import pathlib
from scipy.ndimage import zoom
import tifffile
import tqdm
import typing

from .raw import raw_imread
from .volume import TSVStackBase, TSVVolumeBase, VExtent


def imread(path:pathlib.Path) -> np.ndarray:
    if path.name.endswith(".raw"):
        img = raw_imread(os.fspath(path)).astype(np.float32)
    else:
        img = tifffile.imread(os.fspath(path)).astype(np.float32)
    return img


def zcoord(path:pathlib.Path) -> float:
    """
    Return the putative Z coordinate for a image file path name
    """
    return int(path.name.split(".")[0]) / 10

class ScanStack(TSVStackBase):
    """
    A TSVStack that holds a single piezo travel of paths.
    Within the stack, we assume that the frames are not perfectly aligned,
    but they vary with linear offsets per z in x and y. We make the stack
    look square by trimming a bit off the x and y starts and stops. This
    might be a couple pixels - nothing extreme.
    """
    def __init__(self, x0, y0, z0, z0slice, z1slice, paths):
        super(ScanStack, self).__init__()
        self._x0 = x0
        self._y0 = y0
        self._z0 = z0
        self.x0_trim = 0
        self.x1_trim = 0
        self.y0_trim = 0
        self.y1_trim = 0
        self.x_off_per_z = 0.0
        self.y_off_per_z = 0.0
        self.__paths = paths
        if self.__paths[0].name.endswith(".raw"):
            self.input_plugin = "raw"
        else:
            self.input_plugin = "tiff2D"
        self.z0slice = z0slice
        self.z1slice = z1slice
        self.x_aligned = False
        self.y_aligned = False
        self.z_aligned = False

    @property
    def paths(self):
        return self.__paths

    def x0_getter(self):
        return self._x0 + self.x0_trim

    def x0_setter(self, x0_new):
        self._x0 = int(x0_new) - self.x0_trim
        self.x_aligned = True

    x0 = property(x0_getter, x0_setter)

    @property
    def x1(self):
        self._set_x1y1()
        return self._x0 + self._width  - self.x1_trim

    def y0_getter(self):
        return self._y0 + self.y0_trim

    def y0_setter(self, new_y0):
        self._y0 = int(new_y0) - self.y0_trim
        self.y_aligned = True

    y0 = property(y0_getter, y0_setter)

    @property
    def y1(self):
        return self._y0 + self._height - self.y1_trim

    def z0_getter(self):
        return super(ScanStack, self).z0

    def z0_setter(self, new_z0):
        self._z0 = int(new_z0)
        self.z_aligned = True

    z0 = property(z0_getter, z0_setter)

    def read_plane(self, path:pathlib.Path):
        z = zcoord(path) - self.z0slice
        x_off = int(self.x_off_per_z * z + .5)
        y_off = int(self.y_off_per_z * z + .5)
        img = super(ScanStack, self).read_plane(os.fspath(path))
        x0 = self.x0_trim - x_off
        x1 = img.shape[1] - self.x1_trim + x_off
        y0 = self.y0_trim - y_off
        y1 = img.shape[1] - self.y1_trim + y_off
        return img[y0:y1, x0:x1]

    def as_dict(self) -> dict:
        path0 = self.paths[0]
        orig_z0 = zcoord(path0)
        folder = os.path.split(os.path.dirname(path0))[-1]
        orig_x0, orig_y0 = [float(_) / 10 for _ in folder.split("_")]
        return dict(x0=self.x0,
                    x1=self.x1,
                    y0=self.y0,
                    y1=self.y1,
                    z0=self.z0,
                    z1=self.z1,
                    orig_x0=orig_x0,
                    orig_y0=orig_y0,
                    orig_z0=orig_z0,
                    path=os.path.dirname(path0))


class Edge:
    def __init__(self, xidx, yidx, zidx,
                 xidx1, yidx1, zidx1,
                 score, x_off, y_off, z_off):
        self.xidx = xidx
        self.yidx = yidx
        self.zidx = zidx
        self.xidx1 = xidx1
        self.yidx1 = yidx1
        self.zidx1 = zidx1
        self.score = score
        self.cumulative_score = score
        self.x_off = x_off
        self.y_off = y_off
        self.z_off = z_off
        self.age = None


class AverageDrift:
    """
    The average drift in the x, y and z direction between adjacent x, y and z
    stacks - if the stage axes don't exactly align to the objective axes,
    this will be the chief component in the offsets.
    """
    def __init__(self,
                 xoffx:int, yoffx:int, zoffx:int,
                 xoffy:int, yoffy:int, zoffy:int,
                 xoffz:int, yoffz:int, zoffz:int):
        self.xoffx = xoffx
        self.yoffx = yoffx
        self.zoffx = zoffx
        self.xoffy = xoffy
        self.yoffy = yoffy
        self.zoffy = zoffy
        self.xoffz = xoffz
        self.yoffz = yoffz
        self.zoffz = zoffz

ALIGNMENT_RESULT_T = typing.Tuple[int, typing.Tuple[float, int, int, int]]

class Scanner(TSVVolumeBase):

    def __init__(self,
                 path:pathlib.Path,
                 voxel_size:typing.Tuple[float, float, float],
                 z_stepper_distance=297,
                 piezo_distance=300,
                 z_skip=25,
                 x_slop=5,
                 y_slop=5,
                 z_slop=3,
                 dark=200,
                 drift=None,
                 decimate=1,
                 n_cores=os.cpu_count(),
                 edge_power=1/3):
        """
        Initialize the scanner with the root path to the directory hierarchy
        and the voxel dimensions
        :param path: the path to the root of the hierarchy. It's assumed
        that the first subdirectory level is the X coordinate of the stack,
        in 10ths of microns and the second level is in the form X_Y (again,
        10ths of microns) and the third level is the Z coordinate (in 10ths
        of microns).
        :param voxel_size: The voxel size in microns, x, y and z
        :param z_stepper_distance: the distance in microns of the (alleged)
        travel of the coarse z-stepper as it takes a single step
        :param piezo_distance: the distance in microns of the (alleged) travel
        of the piezo mini-stepper and the big step size of the Z motor.
        :param z_skip: align every z_skip'th plane.
        :param edge_power: Edge scores get multiplied with the scores along
        the chain from the source, making it more difficult to traverse a
        circuitous path from one stack to an adjacent one. We take the last
        edge's cumulative score to the edge_power (between 0 and 1) to
        balance the two concerns.
        """
        self.pool = None
        self.futures_x = {}
        self.futures_y = {}
        self.futures_z = {}
        self.alignments_x = {}
        self.alignments_y = {}
        self.alignments_z = {}
        self._stacks = {}
        self.decimate = decimate
        self.x_voxel_size, self.y_voxel_size, self.z_voxel_size = voxel_size
        self.z_skip = z_skip
        self.x_slop = x_slop
        self.y_slop = y_slop
        self.z_slop = z_slop
        self.dark = dark
        self.edge_power = edge_power
        if drift is None:
            self.drift = AverageDrift(0, 0, 0, 0, 0, 0, 0, 0, 0)
        else:
            self.drift = drift
        self.n_cores = n_cores
        stacks = {}
        for folder in path.iterdir():
            if not folder.is_dir():
                continue
            try:
                x = int(float(folder.name) / self.x_voxel_size / 10)
            except ValueError:
                continue
            for subfolder in folder.iterdir():
                if not subfolder.is_dir():
                    continue
                try:
                    y = int(float(subfolder.name.split("_")[1]) /
                            self.y_voxel_size / 10)
                except:
                    continue
                logging.info("Collecting files for x=%d, y=%d" % (x, y))
                img_paths = sorted(subfolder.glob("*.raw"))
                input_plugin = "raw"
                if len(img_paths) == 0:
                    input_plugin = "tiff2D"
                    img_paths = sorted(subfolder.glob("*.tif*"))
                    if len(img_paths) == 0:
                        continue
                stack_paths = []
                img_path_and_z = sorted(
                    [(int(_.name.rsplit(".", 1)[0]) / 10, _)
                     for _ in img_paths])
                z0 = img_path_and_z[0][0]
                zbase = z0
                current_stack = []
                for z, path in img_path_and_z:
                    if z - z0 >= piezo_distance:
                        stack_paths.append((zbase, current_stack))
                        current_stack = []
                        zbase += z_stepper_distance
                        z0 = z
                    current_stack.append(path)
                stack_paths.append((zbase, current_stack))
                for z, current_stack in stack_paths:
                    z0slice = int(zcoord(current_stack[0]) / self.z_voxel_size)
                    z1slice = int(zcoord(current_stack[-1])/self.z_voxel_size) \
                              + 1
                    stacks[x, y, z] = ScanStack(
                        x, y, int(z / self.z_voxel_size),
                        z0slice, z1slice, current_stack)
        self.xs, self.ys, self.zs = \
            [sorted(set([_[idx] for _ in stacks.keys()]))
                for idx in range(3)]
        for (x, y, z), stack in stacks.items():
            xidx = self.xs.index(x)
            yidx = self.ys.index(y)
            zidx = self.zs.index(z)
            self._stacks[xidx, yidx, zidx] = stack

    @property
    def stacks(self):
        return [list(self._stacks.values())]

    def setup(self, x_slop:int, y_slop:int, z_slop:int, z_skip:int,
              decimate:int, drift:AverageDrift):
        """
        Set up for another round

        :param x_slop: The number of voxels to check in the X direction
        :param y_slop: The number of voxels to check in the Y direction
        :param z_slop: the number of voxels to check in the Z direction
        :param z_skip: Do every z_skip plane in a stack
        :param decimate: the image decimation factor
        :param drift: The calculated mean offsets from the last round
        """
        logging.info("Parameters for next level:")
        logging.info("  x_slop: %d" % x_slop)
        logging.info("  y_slop: %d" % y_slop)
        logging.info("  z_slop: %d" % z_slop)
        logging.info("  decimate: %d" % decimate)
        logging.info("  drift: xx: %d yx: %d zx: %d" %
                     (drift.xoffx, drift.yoffx, drift.zoffx))
        logging.info("         xy: %d yy: %d zy: %d" %
                     (drift.xoffy, drift.yoffy, drift.zoffy))
        logging.info("         xz: %d yz: %d zz: %d" %
                     (drift.xoffz, drift.yoffz, drift.zoffz))

        self.futures_x = {}
        self.futures_y = {}
        self.futures_z = {}
        self.alignments_x = {}
        self.alignments_y = {}
        self.alignments_z = {}
        self.decimate = decimate
        self.z_skip = z_skip
        self.x_slop = x_slop
        self.y_slop = y_slop
        self.z_slop = z_slop
        if drift is None:
            self.drift = AverageDrift(0, 0, 0, 0, 0, 0, 0, 0, 0)
        else:
            self.drift = drift

    def align_all_stacks(self):
        with multiprocessing.Pool(self.n_cores) as self.pool:
            if len(self.xs) > 1:
                self.align_stacks_x()
            if len(self.ys) > 1:
                self.align_stacks_y()
            if len(self.zs) > 1:
                self.align_stacks_z()
            acc = 0
            for src in (self.futures_x, self.futures_y, self.futures_z):
                for k in src:
                    acc += len(src[k])
            bar = tqdm.tqdm(total=acc)
            for src, dest in ((self.futures_x, self.alignments_x),
                              (self.futures_y, self.alignments_y),
                              (self.futures_z, self.alignments_z)):
                for k in src:
                    dest[k] = []
                    for z, future in src[k]:
                        dest[k].append((z, future.get()))
                        bar.update()

    def align_stacks_x(self):
        """
        Align each stack to the one next to it in the X direction
       """
        for xidx, yidx, zidx in itertools.product(
            range(len(self.xs) - 1), range(len(self.ys)), range(len(self.zs))):
            k0 = (xidx, yidx, zidx)
            k1 = (xidx+1, yidx, zidx)
            if k1 not in self._stacks or k0 not in self._stacks:
                continue
            s0 = self._stacks[k0]
            s1 = self._stacks[k1]
            self.futures_x[xidx, yidx, zidx] = self.align_stack_x(s0, s1)

    def align_stacks_y(self):
        """
        Align each stack to the one next to it in the Y direction
        """
        for xidx, yidx, zidx in itertools.product(
            range(len(self.xs)), range(len(self.ys)-1), range(len(self.zs))):
            k0 = (xidx, yidx, zidx)
            k1 = (xidx, yidx+1, zidx)
            if k1 not in self._stacks or k0 not in self._stacks:
                continue
            s0 = self._stacks[k0]
            s1 = self._stacks[k1]
            self.align_stack_y(s0, s1)
            self.futures_y[xidx, yidx, zidx] = self.align_stack_y(s0, s1)

    def align_stacks_z(self):
        """
        Align each stack to the one next to it in the Z direction
        """
        for xidx, yidx, zidx in itertools.product(
            range(len(self.xs)), range(len(self.ys)), range(len(self.zs)-1)):
            k0 = (xidx, yidx, zidx)
            k1 = (xidx, yidx, zidx+1)
            if k1 not in self._stacks or k0 not in self._stacks:
                continue
            s0 = self._stacks[k0]
            s1 = self._stacks[k1]
            self.futures_z[xidx, yidx, zidx] = self.align_stack_z(s0, s1)

    def align_stack_x(self, s0:ScanStack, s1:ScanStack):
        """
        Align stacks that are overlapping in the X direction
        :param s0:
        :type s0:
        :param s1:
        :type s1:
        """
        xc = s1.x0 - s0.x0 + self.drift.xoffx
        x0 = xc - self.x_slop
        x1 = xc + self.x_slop + 1
        y0 = -self.y_slop + self.drift.yoffx
        y1 = self.y_slop + 1 + self.drift.yoffx
        z0m = max(0, self.z_slop + self.drift.zoffx)
        z1m = min(len(s1.paths),
                  len(s1.paths) - self.z_slop + self.drift.zoffx - 1)
        futures = []
        if self.z_skip == "middle":
            zrange = [(z0m + z1m) // 2]
        else:
            zrange = range(z0m, z1m, self.z_skip)
        for z in zrange:
            z0 = z - self.z_slop - self.drift.zoffx
            z1 = z + self.z_slop + 1 - self.drift.zoffx
            futures.append((z, self.pool.apply_async(
                align_one_x,
                (s1.paths[z],
                 s0.paths[z0:z1],
                 x0, x1, y0, y1, z0 - z,
                 self.dark,
                 self.decimate)

            )))
        return futures

    def align_stack_y(self, s0:ScanStack, s1:ScanStack):
        yc = s1.y0 - s0.y0 + self.drift.yoffy
        y0 = yc - self.y_slop
        y1 = yc + self.y_slop + 1
        x0 = -self.x_slop + self.drift.xoffy
        x1 = self.x_slop + 1 + self.drift.xoffy
        z0m = max(0, self.z_slop + self.drift.zoffx)
        z1m = min(len(s1.paths),
                  len(s1.paths) - self.z_slop + self.drift.zoffx - 1)
        futures = []
        if self.z_skip == "middle":
            zrange = [(z0m + z1m) // 2]
        else:
            zrange = range(z0m, z1m, self.z_skip)
        for z in zrange:
            z0 = z - self.z_slop - self.drift.zoffx
            z1 = z + self.z_slop + 1 - self.drift.zoffx
            logging.info("Aligning %s to %s:%s" %
                         (s1.paths[z], s0.paths[z0], s0.paths[z1]))
            logging.info("x0: %d, x1: %d, y0: %d, y1: %d" %
                         (x0, x1, y0, y1))
            futures.append((z, self.pool.apply_async(
                align_one_y,
                (s1.paths[z],
                 s0.paths[z0:z1],
                 x0, x1, y0, y1, z0 - z,
                 self.dark,
                 self.decimate)
            )))
        return futures

    def align_stack_z(self, s0:ScanStack, s1:ScanStack):
        x0 = -self.x_slop + self.drift.xoffz
        x1 = self.x_slop + 1 + self.drift.xoffz
        y0 = -self.y_slop + self.drift.yoffz
        y1 = self.y_slop + 1 + self.drift.yoffz
        s0_paths = s0.paths[-self.z_slop:]
        s1_path = s1.paths[0]
        future = self.pool.apply_async(
                align_one_z, (s0_paths, s1_path, x0, x1, y0, y1,
                              -self.z_slop, self.dark, self.decimate))
        return [[0, future]]

    def compute_median_min_max_without_outliers(self, offs, stds):
        median = np.median(offs)
        off_std = np.std(offs) * stds
        offs = [_ for _ in offs
                if _ >= median - off_std and _ <= median + off_std]
        median = np.median(offs)
        minimum = np.min(offs)
        maximum = np.max(offs)
        return median, minimum, maximum

    def accumulate_offsets(self, alignments, threshold, stds):
        xoffs = []
        yoffs = []
        zoffs = []
        for xidx, yidx, zidx in alignments:
            for z, (score, xoff, yoff, zoff) in alignments[xidx, yidx, zidx]:
                if score < threshold:
                    continue
                xoffs.append(xoff)
                yoffs.append(yoff)
                zoffs.append(zoff)
        xmedian, xmin, xmax = self.compute_median_min_max_without_outliers(
            xoffs, stds)
        ymedian, ymin, ymax = self.compute_median_min_max_without_outliers(
            yoffs, stds)
        zmedian, zmin, zmax = self.compute_median_min_max_without_outliers(
            zoffs, stds
        )
        return (xmedian, xmin, xmax),\
               (ymedian, ymin, ymax),\
               (zmedian, zmin, zmax)

    def calculate_next_round_parameters(self, threshold=.75, stds=3.0,
                                        slop_factor=1.25, z_skip=None):
        (xoffx, xminx, xmaxx), (yoffx, yminx, ymaxx), (zoffx, zminx, zmaxx) = \
        self.accumulate_offsets(self.alignments_x, threshold, stds)
        (xoffy, xminy, xmaxy), (yoffy, yminy, ymaxy), (zoffy, zminy, zmaxy) = \
        self.accumulate_offsets(self.alignments_y, threshold, stds)
        (xoffz, xminz, xmaxz), (yoffz, yminz, ymaxz), (zoffz, zminz, zmaxz) = \
        self.accumulate_offsets(self.alignments_z, threshold, stds)
        x_slop = int(max(xmaxx - xoffx, xoffx - xminx,
                         xmaxy - xoffy, xoffy - xminy,
                         xmaxz - xoffz, xoffz - xminz) *
                     slop_factor) + self.decimate
        y_slop = int(max(ymaxx - yoffx, yoffx - yminx,
                         ymaxy - yoffy, yoffy - yminy,
                         ymaxz - yoffz, yoffz - yminz) *
                     slop_factor) + self.decimate
        z_slop = int(max(zmaxx - zoffx, zoffx - zminx,
                         zmaxy - zoffy, zoffy - zminy,
                         zmaxz - zoffz, zoffz - zminz) *
                     slop_factor) + self.decimate
        drift = AverageDrift(int(xoffx), int(yoffx), int(zoffx),
                             int(xoffy), int(yoffy), int(zoffy),
                             int(xoffz), int(yoffz), int(zoffz))
        self.adjust_stacks(drift, threshold)
        if z_skip is None:
            z_skip = self.z_skip
        self.setup(int(x_slop), int(y_slop), int(z_slop), z_skip,
                   int(self.decimate // 2),
                   AverageDrift(0, 0, 0, 0, 0, 0, 0, 0, 0))

    def adjust_stacks(self, drift:AverageDrift, threshold:float):
        """
        Adjust the stacks according to the calculated alignments

        First, find the highest confidence alignment. Use 1-correlation as the "distance" from one block to
        another and then calculate the shortest path from the highest confidence match. Order blocks by their
        distance from the highest confidence match and update their coordinates relative to the last block on
        the path.

        Since we process in order of distance from the best and since scores are always positive, the block
        to be processed will always be matched with a previously-done block. Also, instead of a simple greedy
        strategy, the paths favor more direct link-ups rather than convoluted paths that can accumulate alignment
        errors.

        :param drift: if the match falls below the threshold, use the drift parameters instead of the
        calculated alignment. The drift is usually pretty accurate.
        :param threshold: The threshold that determines whether to use the calculated parameters or average drift
        """
        for stack in self._stacks.values():
            stack.x_aligned = False
            stack.y_aligned = False
            stack.z_aligned = False

        graph = nx.Graph()
        best_edge = None
        best_score = 0

        d = {}
        node_d = {}
        node_idx_d = {}
        last_node_idx = 0

        def get_node_idx(xidx, yidx, zidx, node_idx):
            key = (xidx, yidx, zidx)
            if key not in node_idx_d:
                node_idx_d[key] = node_idx
                node_d[node_idx] = key
                node_idx += 1
            return node_idx_d[key], node_idx

        for xinc, yinc, zinc, a in (
                (1, 0, 0, self.alignments_x),
                (0, 1, 0, self.alignments_y),
                (0, 0, 1, self.alignments_z)
        ):
            for xidx, yidx, zidx in a:
                xidx1 = xidx + xinc
                yidx1 = yidx + yinc
                zidx1 = zidx + zinc
                z, (score, x_off, y_off, z_off) = a[xidx, yidx, zidx][0]
                if a is self.alignments_z:
                    s0 = self._stacks[xidx, yidx, zidx]
                    z_off += len(s0.paths)
                edge01 = Edge(xidx, yidx, zidx,
                              xidx1, yidx1, zidx1, score,
                              x_off, y_off, z_off)
                edge10 = Edge(xidx1, yidx1, zidx1,
                              xidx, yidx, zidx,
                              score,
                              -x_off, -y_off, -z_off)
                ni0, last_node_idx = get_node_idx(xidx, yidx, zidx, last_node_idx)
                ni1, last_node_idx = get_node_idx(xidx1, yidx1, zidx1, last_node_idx)
                graph.add_edge(ni0, ni1, weight=1 - score + np.finfo(float).eps)
                d[ni0, ni1] = edge01
                d[ni1, ni0] = edge10
                if score > best_score:
                    best_score = score
                    best_edge = edge01
        best_node_idx = node_idx_d[best_edge.xidx, best_edge.yidx, best_edge.zidx]
        score_dict, path_dict = nx.single_source_dijkstra(graph, best_node_idx)
        node_order = sorted(score_dict.keys(),key=lambda k:score_dict[k])
        for node_idx in node_order[1:]:
            path = path_dict[node_idx]
            src_node_idx = path[-2]
            edge = d[src_node_idx, node_idx]
            s0 = self.get_s0_from_edge(edge)
            s1 = self.get_s1_from_edge(edge)
            logging.info("%d,%d,%d->%d,%d,%d" %
                         (edge.xidx, edge.yidx, edge.zidx,
                          edge.xidx1, edge.yidx1, edge.zidx1))
            if edge.score > threshold:
                s1.x0 = s0.x0 + edge.x_off
                s1.y0 = s0.y0 + edge.y_off
                s1.z0 = s0.z0 + edge.z_off
            elif edge.xidx + 1 == edge.xidx1:
                s1.x0 = s0.x0 + drift.xoffx
                s1.y0 = s0.y0 + drift.yoffx
                s1.z0 = s0.z0 + drift.zoffx
            elif edge.xidx == edge.xidx1 + 1:
                s1.x0 = s0.x0 - drift.xoffx
                s1.y0 = s0.y0 - drift.yoffx
                s1.z0 = s0.z0 - drift.zoffx
            elif edge.yidx + 1 == edge.yidx1:
                s1.x0 = s0.x0 + drift.xoffy
                s1.y0 = s0.y0 + drift.yoffy
                s1.z0 = s0.z0 + drift.zoffy
            elif edge.yidx == edge.yidx1 + 1:
                s1.x0 = s0.x0 - drift.xoffy
                s1.y0 = s0.y0 - drift.yoffy
                s1.z0 = s0.z0 - drift.zoffy
            elif edge.zidx + 1 == edge.zidx1:
                s1.x0 = s0.x0 + drift.xoffz
                s1.y0 = s0.y0 + drift.yoffz
                s1.z0 = s0.z0 + drift.zoffz + len(s0.paths)
            elif edge.zidx == edge.zidx1 + 1:
                s1.x0 = s0.x0 - drift.xoffz
                s1.y0 = s0.y0 - drift.yoffz
                s1.z0 = s0.z0 - drift.zoffz - len(s1.paths)
            else:
                raise NotImplementedError("Logic error, nodes are not adjacent")


    def get_s0_from_edge(self, edge:Edge) -> ScanStack:
        return self._stacks[edge.xidx, edge.yidx, edge.zidx]

    def get_s1_from_edge(self, edge: Edge) -> ScanStack:
        return self._stacks[edge.xidx1, edge.yidx1, edge.zidx1]

    def rebase_stacks(self):
        """
        Readjust the stack offsets so that they start at 0, 0, 0
        """
        x0 = np.iinfo(np.int64).max
        y0 = np.iinfo(np.int64).max
        z0 = np.iinfo(np.int64).max
        for stack in self._stacks.values():
            x0 = min(x0, stack.x0)
            y0 = min(y0, stack.y0)
            z0 = min(z0, stack.z0)
        for stack in self._stacks.values():
            stack.x0 = stack.x0 - x0
            stack.y0 = stack.y0 - y0
            stack.z0 = stack.z0 - z0


def align_one_x(tgt_path:pathlib.Path,
                src_paths:typing.Sequence[pathlib.Path],
                x0_off:int, x1_off:int,
                y0_off:int, y1_off:int,
                z_off:int,
                dark:int,
                decimate:int) -> typing.Tuple[float, int, int, int]:
    """
    Align the target to all of the sources, returning the chosen x_offset,
    y_offset and z_offset

    :param tgt_path: The plane to be aligned
    :param src_paths: The choices of planes in the Z direction
    :param x0_off: Start looking at alignment choices in X here
    :param x1_off: End looking here
    :param y0_off: Start looking in Y here
    :param y1_off: End looking here
    :param decimate: Decimate the image by this amount (= zoom by 1/decimate)
    :return: a 4 tuple of the best alignment score, and the x, y and z offsets
    chosen
    """
    best_score, best_xoff, best_yoff, best_zoff = align_one(
        dark, decimate, align_plane_x, src_paths, tgt_path, x0_off,
        x1_off, y0_off, y1_off)
    return best_score, best_xoff, best_yoff, best_zoff + z_off


def align_one(dark, decimate, plane_fn, src_paths, tgt_path, x0_off, x1_off,
              y0_off, y1_off):
    tgt_img = imread(tgt_path)
    src_imgs = [imread(_) for _ in src_paths]
    decimations = []
    d = decimate
    while True:
        decimations.append(d)
        if d == 1:
            break
        d = d // 2
    for decimate in decimations:
        best_score = 0.0
        best_xoff = 0
        best_yoff = 0
        best_zoff = 0
        if decimate != 1:
            tgt_img_decimate = zoom(tgt_img, 1 / decimate)
            src_imgs_decimate = [zoom(_, 1/ decimate) for _ in src_imgs]
        else:
            tgt_img_decimate = tgt_img
            src_imgs_decimate = src_imgs
        for z, src_img in enumerate(src_imgs_decimate):
            best_score, best_xoff, best_yoff, best_zoff = plane_fn(
                best_score, best_xoff, best_yoff, best_zoff, dark,
                decimate, src_img, tgt_img_decimate, x0_off, x1_off, y0_off,
                y1_off, z)
            if best_score  == 0:
                break
            x0_off= best_xoff - decimate
            x1_off = best_xoff + decimate
            y0_off = best_yoff - decimate
            y1_off = best_yoff + decimate
    return best_score, best_xoff, best_yoff, best_zoff


def align_plane_x(best_score, best_xoff, best_yoff, best_zoff, dark,
                  decimate, src_img, tgt_img, x0_off, x1_off, y0_off,
                  y1_off, z):
    for x_off_big in range(x0_off, x1_off, decimate):
        x_off = x_off_big // decimate
        x00 = x_off
        x10 = src_img.shape[1]
        x01 = 0
        x11 = src_img.shape[1] - x_off
        for y_off_big in range(y0_off, y1_off, decimate):
            y_off = y_off_big // decimate
            if y_off > 0:
                y00 = 0
                y10 = tgt_img.shape[0] - y_off
                y01 = y_off
                y11 = tgt_img.shape[0]
            else:
                y00 = -y_off
                y10 = tgt_img.shape[0]
                y01 = 0
                y11 = tgt_img.shape[0] + y_off
            tgt_slice = tgt_img[y01:y11, x01:x11]
            src_slice = src_img[y00:y10, x00:x10]
            mask = (tgt_slice > dark) & (src_slice > dark)
            if np.sum(mask) < np.sqrt(np.prod(tgt_slice.shape)):
                continue
            score = np.corrcoef(tgt_slice.flatten().astype(np.float32),
                                src_slice.flatten().astype(np.float32))[0, 1]
            if score > best_score:
                best_score = score
                best_xoff = x_off_big
                best_yoff = y_off_big
                best_zoff = z
    return best_score, best_xoff, best_yoff, best_zoff


def align_one_y(tgt_path:pathlib.Path,
                src_paths:typing.Sequence[pathlib.Path],
                x0_off:int, x1_off:int,
                y0_off:int, y1_off:int,
                z_off:int,
                dark:int,
                decimate:int) -> typing.Tuple[float, int, int, int]:
    """
    Align the target to all of the sources, returning the chosen x_offset,
    y_offset and z_offset

    :param tgt_path: The plane to be aligned
    :param src_paths: The choices of planes in the Z direction
    :param x0_off: Start looking at alignment choices in X here
    :param x1_off: End looking here
    :param y0_off: Start looking in Y here
    :param y1_off: End looking here
    :param z_slop: the amount of
    :param decimate: Decimate the image by this amount (= zoom by 1/decimate)
    :return: a 4 tuple of the best alignment score, and the x, y and z offsets
    chosen
    """
    best_score, best_xoff, best_yoff, best_zoff =  align_one(
        dark, decimate, align_plane_y,
        src_paths, tgt_path, x0_off, x1_off, y0_off, y1_off)
    return best_score, best_xoff, best_yoff, best_zoff + z_off


def align_plane_y(best_score, best_xoff, best_yoff, best_zoff, dark, decimate,
                  src_img, tgt_img, x0_off, x1_off, y0_off, y1_off, z):
    for x_off_big in range(x0_off, x1_off, decimate):
        x_off = x_off_big // decimate
        if x_off > 0:
            x00 = 0
            x10 = tgt_img.shape[1] - x_off
            x01 = x_off
            x11 = tgt_img.shape[1]
        else:
            x00 = -x_off
            x10 = tgt_img.shape[1]
            x01 = 0
            x11 = tgt_img.shape[1] + x_off
        for y_off_big in range(y0_off, y1_off, decimate):
            y_off = y_off_big // decimate
            y00 = y_off
            y10 = src_img.shape[0]
            y01 = 0
            y11 = src_img.shape[0] - y_off
            tgt_slice = tgt_img[y01:y11, x01:x11]
            src_slice = src_img[y00:y10, x00:x10]
            mask = (tgt_slice > dark) & (src_slice > dark)
            if np.sum(mask) < np.sqrt(np.prod(tgt_slice.shape)):
                continue
            score = np.corrcoef(tgt_slice.flatten().astype(np.float32),
                                src_slice.flatten().astype(np.float32))[0, 1]
            if score > best_score:
                best_score = score
                best_xoff = x_off_big
                best_yoff = y_off_big
                best_zoff = z
    return best_score, best_xoff, best_yoff, best_zoff

def align_one_z(src_paths:typing.Sequence[pathlib.Path],
                tgt_path:pathlib.Path,
                x0:int,
                x1:int,
                y0:int,
                y1:int,
                z_off:int,
                dark:int,
                decimate:int) -> typing.Tuple[float, int, int, int]:
    """
    Align one plane in the x and y direction on behalf of z

    :param src_paths: the paths to the source images
    :param tgt_path: the target image
    :param x0: start at this x offset
    :param x1: end at this x offset
    :param y0:  start at this y offset
    :param y1: end at this y offset
    :param z_off: the z-offset being tested (for bookkeeping only)
    :param dark: For counting minimum # of bright pixels, all values lower
    than this are considered background
    :param decimate: Start out by reducing the size of the image by this factor
    :return: a 4 tuple of the best score,  x offset, y offset and z offset
    """
    best_score, best_x_offset, best_y_offset, best_z_offset =\
        align_one(dark, decimate, align_plane_z, src_paths, tgt_path,
                     x0, x1, y0, y1)
    return best_score, best_x_offset, best_y_offset, best_z_offset + z_off


def align_plane_z(best_score, best_xoff, best_yoff, best_z_off, dark, decimate,
                  src_img, tgt_img, x0, x1, y0, y1,
                  z_off):
    for x_off_big in range(x0, x1, decimate):
        x_off = x_off_big // decimate
        if x_off > 0:
            x00 = 0
            x10 = tgt_img.shape[1] - x_off
            x01 = x_off
            x11 = tgt_img.shape[1]
        else:
            x00 = -x_off
            x10 = tgt_img.shape[1]
            x01 = 0
            x11 = tgt_img.shape[1] + x_off
        for y_off_big in range(y0, y1, decimate):
            y_off = y_off_big // decimate
            if y_off > 0:
                y00 = 0
                y10 = tgt_img.shape[0] - y_off
                y01 = y_off
                y11 = tgt_img.shape[0]
            else:
                y00 = -y_off
                y10 = tgt_img.shape[0]
                y01 = 0
                y11 = tgt_img.shape[0] + y_off
            tgt_slice = tgt_img[y01:y11, x01:x11]
            src_slice = src_img[y00:y10, x00:x10]
            mask = (tgt_slice > dark) & (src_slice > dark)
            if np.sum(mask) < np.sqrt(np.prod(tgt_slice.shape)):
                continue
            score = np.corrcoef(tgt_slice.flatten().astype(np.float32),
                                src_slice.flatten().astype(np.float32))[0, 1]
            if score > best_score:
                best_score = score
                best_xoff = x_off_big
                best_yoff = y_off_big
                best_z_off = z_off
    return best_score, best_xoff, best_yoff, best_z_off


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    root = "/mnt/cephfs/users/lee/data/tsv-scan"
    voxel_size = [.41, .41, 2.0]
    drift = AverageDrift(0, 0, 0, 0, 0, 0, 0, 0, 0)
    scanner = Scanner(
        pathlib.Path(root),
        voxel_size,
        z_skip="middle",
        x_slop=30,
        y_slop=30,
        z_slop=6,
        decimate=8,
        dark=300,
        drift=drift)


    def dump_round(fd):
        json.dump(dict(
            x=dict([(",".join([str(_) for _ in k]), scanner.alignments_x[k])
                    for k in scanner.alignments_x]),
            y=dict([(",".join([str(_) for _ in k]), scanner.alignments_y[k])
                    for k in scanner.alignments_y]),
            z=dict([(",".join([str(_) for _ in k]), scanner.alignments_z[k])
                    for k in scanner.alignments_z])), fd, indent=2)

    def load_round(scanner, fd):
        d = json.load(fd)
        scanner.alignments_x = \
            dict([(tuple(int(_) for _ in k.split(",")), d["x"][k])
                  for k in d["x"]])
        scanner.alignments_y = \
            dict([(tuple(int(_) for _ in k.split(",")), d["y"][k])
                  for k in d["y"]])
        scanner.alignments_z = \
            dict([(tuple(int(_) for _ in k.split(",")), d["z"][k])
                  for k in d["z"]])


    if not os.path.exists("/tmp/round1.json"):
        scanner.align_all_stacks()
        with open("/tmp/round1.json", "w") as fd:
            dump_round(fd)
    else:
        with open("/tmp/round1.json", "r") as fd:
            load_round(scanner, fd)
    scanner.calculate_next_round_parameters()
    scanner.rebase_stacks()
    for z in tqdm.tqdm(range(scanner.volume.z0, scanner.volume.z1)):
        plane = scanner.imread(VExtent(0, scanner.volume.x1,
                                       0, scanner.volume.y1,
                                       z, z+1), np.uint16)
        path = "/mnt/cephfs/users/lee/data/tsv-scan/stitched/img_%04d.tiff" % z
        tifffile.imsave(path, plane.reshape(plane.shape[1], plane.shape[2]),
                        compress=3)
