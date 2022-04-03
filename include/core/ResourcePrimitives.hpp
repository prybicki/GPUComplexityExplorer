#pragma once

#include <types/count_t.hpp>
#include <typeindex>
#include <array>
#include <vector_types.h>

#include <data/Type.hpp>

struct ThreadsLayout
{
	// TODO: this is ambiguous
	ThreadsLayout(count_t launchDims, count_t blockDims=256);
	// ThreadLayout(std::array<count_t, 1> launchDims, std::array<count_t, 1> blockDims={256});
	ThreadsLayout(std::array<count_t, 2> launchDims, std::array<count_t, 2> blockDims={16, 16});
	ThreadsLayout(std::array<count_t, 3> launchDims, std::array<count_t, 3> blockDims={16, 16, 16});


private:
	ThreadsLayout(dim3 gridDim, dim3 blockDim) : gridDim(gridDim), blockDim(blockDim) {}
	dim3 gridDim;
	dim3 blockDim;
	friend struct ResourceManager;
};

struct DeviceMemory
{
private:
	DeviceMemory(void* ptr, count_t elemCount, Type valueType) : ptr(ptr), elemCount(elemCount), valueType(valueType) {}

private:
	void* ptr;
	count_t elemCount;
	Type valueType;

	friend struct ResourceManager;
};

using memory_t = std::shared_ptr<DeviceMemory>;