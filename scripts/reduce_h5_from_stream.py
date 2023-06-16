import numpy as np
import h5py
import hdf5plugin
import argparse
import os
from crystfelparser.crystfelparser import streamfile_parser
from tqdm import tqdm

def get_frame(frame_idx, h5_file):
    with h5py.File(h5_file, 'r') as f:
        data = f['/entry/data/data'][frame_idx]
    return data

def save_frames_to_h5(frames, output_file):
    with h5py.File(output_file, 'w') as f:
        grp = f.create_group('/entry/data')
        # Use the hdf5plugin to specify the Bitshuffle and LZ4 filters
        grp.create_dataset('data', data=frames, **hdf5plugin.Bitshuffle())

def process_files(h5_file, stream_file, output_file, window_size):
    stream = streamfile_parser(stream_file)
    indexable_frames = stream.get_indexable_frames()

    new_frames = []
    for idx in tqdm(indexable_frames):
        original_frame = get_frame(idx, h5_file)
        new_frame = np.zeros_like(original_frame)
        for point in stream.parsed[idx]['predicted_reflections'][:,:2]:
            x, y = point.astype(int)
            new_frame[x-window_size:x+window_size+1, y-window_size:y+window_size+1] = \
                original_frame[x-window_size:x+window_size+1, y-window_size:y+window_size+1]
        new_frames.append(new_frame)

    new_frames = np.array(new_frames)
    save_frames_to_h5(new_frames, output_file)

def main():
    parser = argparse.ArgumentParser(description='Process an h5 file and a stream file.')
    parser.add_argument('--h5file', type=str, required=True,
                        help='Input h5 file')
    parser.add_argument('--streamfile', type=str, required=True,
                        help='Input stream file')
    parser.add_argument('--output', type=str, 
                        help='Output h5 file', default=None)
    parser.add_argument('--window_size', type=int,
                        help='Window size for each spot. Default: 15', default=15)

    args = parser.parse_args()

    # If output file is not given, generate one based on the stream file name
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.streamfile))[0]
        args.output = base_name + "_sparse.h5"

    process_files(args.h5file, args.streamfile, args.output, args.window_size)

if __name__ == "__main__":
    main()
