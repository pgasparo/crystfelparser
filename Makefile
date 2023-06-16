SHELL := /bin/bash

CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

all: install

install:
	@ ($(SHELL) INSTALL.sh)
	@ ($(CONDA_ACTIVATE) crystfelparser ; python setup.py install)
