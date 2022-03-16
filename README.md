# crystfelparser
![](docs/crystfelexporter_schema.png)

![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)

## Overview 
The crystfelparser application is an easy-to-use, open-source toolbox for parsin the output stream from indexamajig.

## Installation

Linux and MacOS are fully supported. Windows too, through WSL. 

### Installation via conda

We recommend and support installation via the [conda](https://docs.conda.io/en/latest/miniconda.html) package manager, and that a fresh environment is created beforehand. Then fetch the package from our channel:

```bash
conda install crystfelparser -c crystfelparser -c conda-forge
```

### Installation via Docker

A CUDA-enabled container can be pulled from [DockerHub](https://hub.docker.com/tobedone). 

We also provide a Dockerfile for manual builds:

```bash
docker build -t crystfelparser . 
```

Attach to the provided container with:

```bash
docker run -it crystfelparser bash
```

## Quick start

Bla bla.

## Tutorials

In-depth tutorials can be found in the `tutorials` subfolder. These include: 

- [something.ipynb](tutorials/something.ipynb): This showcases...
- [something_else.ipynb](tutorials/something_else.ipynb): This dives into...
