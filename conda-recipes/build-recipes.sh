#!/bin/bash

case "$1" in
    linux-32)
	miniconda_url=http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh
        ;;
    linux-64)
	miniconda_url=http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
        ;;
    *)
        echo $"Usage: $0 [linux-32|linux-64]"
        exit 1
esac

wget $miniconda_url -O miniconda.sh
bash miniconda.sh -b -p $PWD/miniconda
export PATH=$PWD/miniconda/bin:$PATH
conda config --add channels yasserglez
conda config --set binstar_upload yes
conda update conda
conda install conda-build
conda install patchelf
conda install binstar

conda build fim
conda build fann
conda build fann2
conda build igraph
conda build python-igraph
