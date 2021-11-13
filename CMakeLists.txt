cmake_minimum_required(VERSION 3.16)

# Enable C++20 in CUDA device code (supported only by clang)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_COMPILER clang++-12)
set(CMAKE_CUDA_COMPILER clang-12)
project(THC LANGUAGES CXX CUDA)

# Configure and compile Magnum
include(cmake/ConfigureMagnum.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

# Enable all warnings in Debug mode
if (CMAKE_BUILD_TYPE EQUAL "Debug")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CUDA_FLAGS "-Wall -Wextra")
endif()

# Find dependencies
find_package(FMT)
find_package(CUDAToolkit REQUIRED)
find_package(Magnum REQUIRED GL GlfwApplication Trade AnyImageImporter Shaders MeshTools Primitives)

# ---------------- Define project artifacts ---------------- #

add_library(GPUKernels src/kernels.cu)
target_include_directories(GPUKernels PRIVATE include)

add_executable(THC src/main.cpp)
target_include_directories(THC PRIVATE include ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(THC
	# Internal modules
	GPUKernels
	# Magnum modules
	Magnum::Application
	Magnum::GL
	Magnum::Magnum
	Magnum::Trade
	Magnum::AnyImageImporter
	Magnum::MeshTools
	Magnum::Shaders
	Magnum::Primitives
	MagnumPlugins::StbImageImporter
	# Other dependencies
	fmt::fmt
	${CUDA_CUDART}
)

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
	# Clang uses special mangling for device kernel names (adding __device_stub__ prefix)
	# This makes it impossible to obtain kernel address from .cpp translation unit.
	# To fix this, all project files needs to be compiled as if they were cuda files,
	# since then clang is generates correct relocations in .cpp files.
	target_compile_options(THC PRIVATE --language=cuda --cuda-host-only)
endif()
