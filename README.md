# GPUComplexityExplorer

GPUComplexityExplorer is toolkit (library) for an amateur computational experimentation done purely on GPU using CUDA and OpenGL interop (via [Magnum Engine](https://magnum.graphics/)).

A short roadmap:
- Evolution of Cellular Automata

## Building

Building instructions assume Ubuntu 20.04 or compatible OS.

### Prerequisites

- CUDA 11
```
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
```
- CMake 3.18
```bash
# WARNING: This will remove exisiting CMake installation and dependent packages. Proceed with caution!
sudo apt purge --auto-remove cmake
sudo apt install wget gpg software-properties-common
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
source /etc/os-release
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main"
sudo apt update
sudo apt install cmake     
```

- Clang 12
- GLFW3
- {FMT}
```bash
sudo apt install clang-12 libglfw3-dev libfmt-dev
```

- Magnum Graphics Engine sources
```bash
sudo pip3 install vcstool
vcs import --repos --shallow --recursive < external.repos
```

### Building

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
