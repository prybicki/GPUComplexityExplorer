# CMake 3.17 provides find_package(CUDAToolkit REQUIRED)
cmake_minimum_required(VERSION 3.17)

# Enable C++20 in CUDA device code (supported only by clang)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_COMPILER clang++-12)
set(CMAKE_CUDA_COMPILER clang-12)
project(THC LANGUAGES CXX CUDA)

# Configure and compile Magnum
include(cmake/ConfigureMagnum.cmake)

# Enable all warnings in Debug mode
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CUDA_FLAGS "-Wall -Wextra")
endif()

# Find dependencies
find_package(FMT)
find_package(CUDAToolkit REQUIRED)
find_package(Magnum REQUIRED GL GlfwApplication Trade AnyImageImporter Shaders MeshTools Primitives DebugTools)
find_package(MagnumIntegration REQUIRED ImGui)
find_package(Corrade REQUIRED Utility)
set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

# ---------------- Define project artifacts ---------------- #

include_directories(include)

# Add resources compiled into the binary (e.g. shaders)
corrade_add_resource(ShaderResources shaders/shaders.conf)

# RAII Wrapper for ShaderResources
add_library(ShaderSources src/ShaderSources.cpp ${ShaderResources})
target_link_libraries(ShaderSources Corrade::Utility)

add_library(GPUKernels src/cuda/kernels.cu)
#set_property(TARGET GPUKernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_library(THC
	src/ResourceManager.cpp
	src/Visualizer.cpp
	src/Field2DShader.cpp
)
target_include_directories(THC PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(THC
	# Magnum modules
	Magnum::Application
	Magnum::GL
	Magnum::Magnum
	Magnum::Trade
	Magnum::MeshTools
	Magnum::Shaders
	Magnum::Primitives
	Magnum::DebugTools
	MagnumIntegration::ImGui
#	MagnumPlugins::StbImageImporter
	# Other dependencies
	fmt::fmt
	CUDA::cudart
	CUDA::curand
	ImPlot
	GPUKernels
	ShaderSources
)

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
	# Clang uses special mangling for device kernel names (adding __device_stub__ prefix)
	# This makes it impossible to obtain kernel address from .cpp translation unit.
	# To fix this, all project files needs to be compiled as if they were cuda files,
	# since then clang is generates correct relocations in .cpp files.
	target_compile_options(THC PUBLIC --language=cuda --cuda-host-only)
endif()

add_subdirectory(apps)
