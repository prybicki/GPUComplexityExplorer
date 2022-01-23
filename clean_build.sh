#!/bin/bash

rm -rf build/
rm -rf external/
vcs import --repos --shallow --recursive < external.repos
mkdir build
cmake -B build -S .
make -C build -j $(nproc)

