# crystfelparser
![Schema](docs/crystfelexporter_schema.png)

![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)

## Overview 
The crystfelparser application is an easy-to-use, open-source toolbox for parsing the output stream from [indexamajig](https://www.desy.de/~twhite/crystfel/manual-indexamajig.html).
Basically, using this tool you can transform a raw text file into series of dictionaries, where for each frame you havea kewyword and the corresponding value, e.g. the strong reflections found by the spot finder and the positions of the predicted Bragg's reflections where the frames are indexable.

This library can also be used to parse the outputs from [XDS](https://xds.mr.mpg.de/html_doc/xds_files.html), i.e. [XDS_ASCII.HKL](https://xds.mr.mpg.de/html_doc/xds_files.html#XDS_ASCII.HKL) and [SPOT.XDS](https://xds.mr.mpg.de/html_doc/xds_files.html#SPOT.XDS).

## Installation

Linux and MacOS are fully supported. Windows too, through WSL. 

### Manual installation

```bash
git clone https://github.com/pgasparo/crystfelparser 
cd crystfelparser && make
```

Alternatively, if you already have a workin python environment with all the necessary libraries:

```bash
git clone https://github.com/pgasparo/crystfelparser 
cd crystfelparser && pip install . 
```


### Installation via Docker

A Dockerfile for manual builds is provided:

```bash
docker build -t crystfelparser . 
```

Attach to the provided container with:

```bash
docker run -it crystfelparser bash
```

You are now inside the container and you can use the script.

## Quick start: parsing indexamajig's output stream

Usage from the command line, as a script, is simple: just specify an input stream and (optional) an output file.

```bash
crystfelparser --stream ../tutorials/crystfel.stream
```

To import and use functions from the library in your code:

```python
from crystfelparser.crystfelparser import streamfile_parser

# parse a stream file
tutorial_stream = streamfile_parser("tutorials/crystfel.stream")
len(tutorial_stream.parsed[25])
# Output: 13
```

To load a previously saved h5 file:

```python
from crystfelparser.utils import load_dict_from_hdf5

# parse a stream file
parsed=load_dict_from_hdf5("parsed_stream.h5")
len(parsed[25])
# Output: 13
```

### Parsing XDS output

XDS outputs can be parsed using the command line tool `xdsparser`:

```bash
xdsparser --file XDS_ASCII.HKL --dt 3
```

or if you want to read the ouput from colspot

```bash
xdsparser --file SPOT.XDS --dt 0 --colspot
```

## Tutorials

In-depth tutorials can be found in the `tutorials` subfolder. These include: 

- [something.ipynb](tutorials/something.ipynb): This showcases...
- [something_else.ipynb](tutorials/something_else.ipynb): This dives into...
