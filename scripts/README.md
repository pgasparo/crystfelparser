# H5 Stream File Processor

This Python script processes h5 and stream files from crystalline diffraction data, creating a new h5 file with sparse representation of data and applying bittshuffle + lz4 compression. There's another script that creates a CXI file from a CrystFEL stream file and linked HDF5 image data.

## Dependencies

- Python 3.x
- numpy
- h5py
- hdf5plugin
- crystfelparser
- tqdm
- argparse

You can install all Python dependencies by running:

```bash
pip install numpy h5py hdf5plugin tqdm argparse
```

## Scripts

### reduce_h5_from_stream.py

This script compresses raw h5 files from streams.

Command line arguments for the script are as follows:

```bash
python reduce_h5_from_stream.py --h5file <h5_input_file> --streamfile <stream_input_file> [--output <output_file>] [--window_size <window_size>]
```
- `--h5file`: (Required) Input h5 file path.
- `--streamfile`: (Required) Input stream file path.
- `--output`: (Optional) Output h5 file path. If not provided, an output file name will be generated based on the stream file name with the pattern <basename>_sparse.h5.
- `--window_size`: (Optional) Window size for each spot. Default is 15.

### cxi_peaks_list.py

This script writes a CXI file with peaks_list from a CrytFEL stream file, and linked HDF5 image data.

Command line arguments for the script are as follows:

```bash
python cxi_peaks_list.py <streamfile> <h5filename> [--output <output_filename>] [--datapath <source_HDF5_path_in_linked_h5filename>]
```
- `streamfile`: (Required) The CrytFEL stream file path.
- `h5filename`: (Required) The HDF5 image data file path.
- `--output`: (Optional) The CXI output filename. Default is `<h5filename>.cxi`.
- `--datapath`: (Optional) Source HDF5 path in linked h5filename. Default is `/entry/data/data`.
