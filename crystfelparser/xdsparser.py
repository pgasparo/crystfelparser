"""Console script for crystfelparser."""

from collections import defaultdict
from xmlrpc.client import Boolean
from tqdm import tqdm
import numpy as np
import argparse
import sys

from crystfelparser.utils import save_dict_to_hdf5

# Just round up and down the time, we can extend this to a larger time window


def round_time(raw_spots, dt=0):
    """ 
    Function for rouding the time/angle

    Args:
      raw_spots: matrix where each row is [SPOT_X, SPOT_Y, FRAME]
      dt: time window to flatten into a single frame [-dt,+dt]

    Returns:
      A numpy array where each row is [SPOT_X, SPOT_Y, FRAME_ROUNDED]
    """

    # we could remove duplicates here
    # set_of_floats=set(list(map(tuple,raw_spots[:,:3])))
    max_frame = int(np.max(raw_spots[:, 2]))
    expanded_list = []
    for spot in raw_spots:
        # round to the time to the earest integer
        rounded = np.round(spot[2])
        # set the lower bound
        min_round = max(0, rounded - dt)
        # set the upper bound
        max_round = min(rounded + dt + 1, max_frame + 1)

        for new_time in np.arange(min_round, max_round):
            expanded_list.append((spot[0], spot[1], new_time))

    return np.asarray(expanded_list)


def match_spots_frame_expanded(spots, n_frame):
    """
    Given the list of spots found by XDS and a specific frame number, 
    return the list of centers found matching the frame number given

    Args:
      spots: matrix where each row is [SPOT_X, SPOT_Y, FRAME]
      n_frame: the frame to select

    Returns:
      A numpy array containing the spots for the selected frame
    """

    # find all the spots in the same frame
    idx_rows = np.where(list(map(int, spots[:, 2])) == n_frame)
    tmp_list = spots[idx_rows, ([0], [1])].T

    # I use set to remove possible duplicates
    return np.asarray(list(set(list(map(tuple, tmp_list)))))


def get_list_spots_fromfile(filename, colspot=False, timerounded=True, shift_min=True, dt=3):
    """
    Load spot centers from a file and return a dictionary where each key
    is a frame number

    Args:
      filename: filepath containing the output from XDS [e.g. SPOT.XDS, XDS.XDS_ASCII.HKL]
      colspot: flag to specify if the file is XDS.XDS_ASCII.HKL or SPOT.XDS [Default: False]
      timerounded: flag to round or not the frame number [default: True]
      shift_min: flag to shif the minimum frame to zero [default: True]
      dt: window of frames to collaps within a central frame [-dt, dt] [default: 3]

    Returns:
      A dictionary with the spots per frame, where each frame is a frame number
    """

    dict_spots = defaultdict(lambda: [])

    if colspot:
        # reading SPOT.XDS
        spots_reflections_raw = np.loadtxt(filename)
    else:
        # eading XDS_ASCII.HKL
        spots_reflections_raw = []
        with open(filename, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.split()[0][0] != "!":
                    spots_reflections_raw.append(
                        list(map(float, line.split()[5:8])))
        spots_reflections_raw = np.asarray(spots_reflections_raw)

    if timerounded:
        spots_reflections_raw = round_time(spots_reflections_raw, dt)

    # get unique frames
    # frames=set(rounded_spots[:,2])
    frames = set(spots_reflections_raw[:, 2].astype(int))
    min_framenum = 0
    if shift_min:
        min_framenum = min(frames)

    for fr in tqdm(frames):
        dict_spots[fr -
                   min_framenum] = match_spots_frame_expanded(spots_reflections_raw, fr)

    return dict_spots

###################################################
# Script to be called from the command line


def parse_args():
    """Parser"""
    parser = argparse.ArgumentParser(
        description='Console script to parse XDS_ASCII.HKL (or SPOT.XDS)')

    parser.add_argument(
        '--file', default="XDS_ASCII.HKL", help='XDS file to parse [e.g. XDS_ASCII.HKL]'
    )

    parser.add_argument(
        '--colspot', type=Boolean, default=False, help='Reading a SPOT.XDS file (COLSPOT output) [Default: False]'
    )

    parser.add_argument(
        '--dt', type=int, default=0, help='Time window to flatten [-dt, dt] [Default: 0]'
    )

    parser.add_argument(
        '--output',
        default="parsed_xds.h5",
        help='Parsed file, stored in hdf5 format [default: parsed_xds.h5]',
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
    else:
        args = parser.parse_args()
        # args.func(**vars(args))

        return args


def main():
    """ """
    # read from the parser
    inputs = parse_args()

    # parse the input stream
    print("Parsing {}".format(inputs.file))
    if inputs.dt > 0:
        timerounded = True

    parsed_stream = get_list_spots_fromfile(
        inputs.file, inputs.colspot, timerounded, True, inputs.dt)

    # save the input stream to a .h5 file
    save_dict_to_hdf5(parsed_stream, inputs.output)
    print("Parsed frames saved in {}".format(inputs.output))


if __name__ == "__main__":
    sys.exit(main())
