#!/usr/bin/env python
import os
import argparse
import h5py
import hdf5plugin
import numpy as np
from crystfelparser.crystfelparser import stream_to_dictionary

H5_DATAPATH = "/entry/data/data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Writes a CXI file with peaks_list from a CrytFEL stream file, and linked HDF5 image data"
    )
    parser.add_argument("streamfile", type=str, help="The CrytFEL stream file")
    parser.add_argument("h5filename", type=str, help="The HDF5 image data")
    parser.add_argument(
        "--output",
        default="",
        type=str,
        help="The CXI output filename. Default: <h5filename>.cxi",
    )
    parser.add_argument(
        "--datapath",
        default=H5_DATAPATH,
        type=str,
        help=f"Source HDF5 path in linked h5filename. Default: {H5_DATAPATH}",
    )
    return parser.parse_args()


def stack_uneven(arrays, max_sizes=None, dtype=np.float32):
    """
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            max_sizes (float, optional):

    Returns:
            np.ndarray
    """
    sizes = [a.shape for a in arrays]
    if max_sizes is None:
        max_sizes = np.max(list(zip(*sizes)), -1)
    # The resultant array is stacked on the first dimension
    result = np.zeros((len(arrays),) + tuple(max_sizes), dtype=dtype)
    for i, a in enumerate(arrays):
        # The shape of this array `a`, turned into slices
        slices = tuple(slice(0, s) for s in sizes[i])
        # Overwrite a block slice of `result` with this array `a`
        result[i][slices] = a
    return result


if __name__ == "__main__":
    args = parse_args()
    if not args.output:
        args.output = os.path.splitext(args.h5filename)[0] + ".cxi"

    parsed = stream_to_dictionary(args.streamfile)
    peaks = [[]] * len(parsed)
    npeaks = [[]] * len(parsed)
    for i in range(len(parsed)):
        frame = parsed[i]
        npeaks[i] = frame["num_peaks"]
        peaks[i] = frame["peaks"]
    peaks = stack_uneven(peaks, max_sizes=(1024, 4))

    # open the HDF5 CXI file for writing
    with h5py.File(args.output, "w") as f:
        f[args.datapath] = h5py.ExternalLink(args.h5filename, args.datapath)

        entry_1 = f.create_group("entry_1")
        result_1 = entry_1.create_group("entry_1")

        # populate the file with the classes tree
        result_1.create_dataset("nPeaks", data=npeaks)
        result_1.create_dataset("peakXPosRaw", data=peaks[..., 0])
        result_1.create_dataset("peakYPosRaw", data=peaks[..., 1])
        result_1.create_dataset("peakTotalIntensity", data=peaks[..., 3])
