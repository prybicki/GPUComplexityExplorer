# TinyHadronCollider

## Building

Building instructions assume Ubuntu 20.04 or compatible OS.

#### Prerequisites

This project requires a valid CUDA installation: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
Other dependencies can be downloaded/installed by running the following command **in project directory**:

```bash
sudo apt install cmake clang-12 libglfw3-dev libfmt-dev
sudo pip3 install vcstool
vcs import --repos --shallow --recursive --skip-existing < external.repos
```

#### Building

```bash
mkdir build
cd build
cmake ../
make -j $(nproc)
```

To verify build:

```
./THC
```
