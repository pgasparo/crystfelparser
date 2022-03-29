"""Console script for crystfelparser."""

from collections import defaultdict
import numpy as np
import argparse
import sys

from crystfelparser.utils import save_dict_to_hdf5

###################################################
# Script to be called from the command line


def parse_args():
    """Parser"""
    parser = argparse.ArgumentParser(
        description='Console script to parse XDS.ASCII')

    parser.add_argument(
        '--stream', default="XDS.ASCII", help='XDS file to parse [e.g. XDS.ASCII]'
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
    print("Parsing {}".format(inputs.stream))
    # parsed_stream = stream_to_dictionary(inputs.stream)

    # save the input stream to a .h5 file
    # dictionary_parsed_to_h5(parsed_stream, inputs.output)
    # print("Indexable frames saved in {}".format(inputs.output))


if __name__ == "__main__":
    sys.exit(main())
