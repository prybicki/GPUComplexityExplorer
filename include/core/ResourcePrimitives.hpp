#pragma once

#include <types/count_t.hpp>
#include <typeindex>
#include <array>
#include <vector_types.h>

#include <data/Type.hpp>
#include <compute/Accessors.hpp>

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
	friend struct MemoryManager;
};

struct DeviceMemory
{
	~DeviceMemory();
	template<typename R>
	LinMemAcc2D<R> accessor2D(Vec2c dims) {
		// TODO type and bound checking
		return {reinterpret_cast<R*>(ptr), dims};
	}
private:
	DeviceMemory(void* ptr, count_t elemCount, Type valueType) : ptr(ptr), elemCount(elemCount), valueType(valueType) {}


private:
	void* ptr;
	count_t elemCount;
	Type valueType;

	friend struct MemoryManager;
};

using dev_mem_t = std::shared_ptr<DeviceMemory>;