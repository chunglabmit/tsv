"""convert.py - programs to convert stacks to output formats"""

import argparse
import multiprocessing
import numpy as np
import os
import tifffile
from .volume import VExtent, TSVVolume
import tqdm


def convert_to_2D_tif(v, output_pattern,
                      mipmap_level=None,
                      volume=None,
                      dtype=None,
                      silent=False,
                      compression=4,
                      cores=multiprocessing.cpu_count(),
                      ignore_z_offsets=False):
    """Convert a terastitched volume to TIF

    :param v: the volume to convert
    :param output_pattern: File naming pattern. output_pattern.format(z=z) is
         called to get the path names for each TIF plane. The directory must
         already exist.
    :param mipmap_level: mipmap decimation level, e.g. "2" to output files
         at 1/4 resolution.
    :param volume: an optional VExtent giving the volume to output
    :param dtype: an optional numpy dtype, defaults to the dtype indicated
                  by the bit depth
    :param cores: # of processes to run simultaneously
    :param ignore_z_offsets: True to ignore the Z offsets in the xml file
    """
    if volume is None:
        volume = v.volume
    if dtype is None:
        dtype = v.dtype
    if mipmap_level is not None:
        decimation = 2 ** mipmap_level
    else:
        decimation = 1
    futures = []
    with multiprocessing.Pool(cores) as pool:
        for z in range(volume.z0, volume.z1, decimation):
            futures.append(pool.apply_async(
                convert_one_plane,
                (v, compression, decimation, dtype, output_pattern,
                 volume, z)))
        for future in tqdm.tqdm(futures):
            future.get()


def convert_one_plane(v, compression, decimation, dtype,
                      output_pattern, volume, z):

    mini_volume = VExtent(
        volume.x0, volume.x1, volume.y0, volume.y1, z, z + 1)
    plane = v.imread(mini_volume, dtype)[0]
    if decimation > 1:
        plane = plane[::decimation, ::decimation]
    path = output_pattern.format(z=z)
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    tifffile.imsave(path, plane, compress=compression)


def make_diag_stack(xml_path, output_pattern,
                    mipmap_level=None,
                    volume=None,
                    dtype=None,
                    silent=False,
                    compression=4,
                    cores=multiprocessing.cpu_count()):
    v = TSVVolume.load(xml_path)
    if volume is None:
        volume = v.volume
    if dtype is None:
        dtype = v.dtype
    if mipmap_level is not None:
        decimation = 2 ** mipmap_level
    else:
        decimation = 1
    if cores == 1:
        for z in tqdm.tqdm(range(volume.z0, volume.z1, decimation)):
            make_diag_plane(v, compression, decimation, dtype, mipmap_level,
                            output_pattern,volume, z)
        return

    futures = []
    with multiprocessing.Pool(cores) as pool:
        for z in range(volume.z0, volume.z1, decimation):
            futures.append(pool.apply_async(
                make_diag_plane,
                (v, compression, decimation, dtype, mipmap_level,
                 output_pattern, volume, z)))
        for future in tqdm.tqdm(futures):
            future.get()


def make_diag_plane(v, compression, decimation, dtype, mipmap_level, output_pattern, volume, z):

    mini_volume = VExtent(
        volume.x0, volume.x1, volume.y0, volume.y1, z, z + 1)
    plane = v.make_diagnostic_img(mini_volume)[0].astype(dtype)
    if plane.shape[2] > 3:
        plane = plane[:, :, :3]
    if mipmap_level is not None:
        plane = plane[::decimation, ::decimation]
    if plane.shape[2] < 3:
        plane = np.dstack(
            list(plane.transpose(2, 0, 1)) +
            [np.zeros(plane.shape[:2], plane.dtype)] * (3-plane.shape[2]))
    path = output_pattern.format(z=z)
    tifffile.imsave(path, plane, compress=compression, photometric="rgb")


def main():
    parser = argparse.ArgumentParser(
        description="Make a z-stack out of a Terastitcher volume"
    )
    args, mipmap_level, volume = parse_args(parser)
    v = TSVVolume.load(args.xml_path, args.ignore_z_offsets, args.input)

    convert_to_2D_tif(v,
                      args.output_pattern,
                      mipmap_level=mipmap_level,
                      volume=volume,
                      silent=args.silent,
                      compression=args.compression,
                      cores=args.cpus,
                      ignore_z_offsets=args.ignore_z_offsets)


def parse_args(parser:argparse.ArgumentParser):
    """Standardized argument parser for convert functions

    :param parser: an argument parser, possibly configured for the application
    :returns: the parsed argument dictionary, the mipmap level and the volume \
    (or None for the entire volume)
    """
    parser.add_argument(
        "--xml-path",
        required=True,
        help="Path to the XML file generated by Terastitcher")
    parser.add_argument(
        "--output-pattern",
        required=True,
        help='Pattern for tif files, e.g. "output/img_{z:04d}.tif"')
    parser.add_argument(
        "--mipmap-level",
        default=0,
        type=int,
        help="Image decimation level, e.g. --mipmap-level=2 means 4x4x4 "
             "smaller image")
    parser.add_argument(
        "--volume",
        default="",
        help='Volume to be captured. Format is "<x0>,<x1>,<y0>,<y1>,<z0>,<z1>".'
             ' Default is whole volume.')
    parser.add_argument(
        "--compression",
        default=4,
        type=int,
        help="TIFF compression level (0-9, default=3)")
    parser.add_argument(
        "--silent",
        action="store_true")
    parser.add_argument(
        "--cpus",
        default=multiprocessing.cpu_count(),
        type=int,
        help="Number of CPUs to use for multiprocessing")
    parser.add_argument(
        "--ignore-z-offsets",
        action="store_true",
        help="Ignore any z offsets in the stitching XML file."
    )
    parser.add_argument(
        "--input",
        help="Optional input location for unstitched stacks. Default is to "
        "use the value encoded in the --xml-path file"
    )

    args = parser.parse_args()
    if args.mipmap_level == 0:
        mipmap_level = None
    else:
        mipmap_level = args.mipmap_level
    if args.volume != "":
        x0, x1, y0, y1, z0, z1 = map(int, args.volume.split(","))
        volume = VExtent(x0, x1, y0, y1, z0, z1)
    else:
        volume = None
    return args, mipmap_level, volume


def diag():
    """Produce a diagnostic image"""
    parser = argparse.ArgumentParser(
        description = "Make a false-color diagnostic image stack"
    )
    args, mipmap_level, volume = parse_args(parser)
    make_diag_stack(args.xml_path,
                    args.output_pattern,
                    mipmap_level=mipmap_level,
                    volume=volume,
                    silent=args.silent,
                    compression=args.compression,
                    cores=args.cpus)


if __name__ == "__main__":
    import os
    if os.environ.get("TSV_DIAG", False):
        diag()
    else:
        main()