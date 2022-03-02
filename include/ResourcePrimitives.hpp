#pragma once

#include <aliases.hpp>
#include <typeindex>
#include <array>
#include <vector_types.h>

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

struct TypeInfo
{
	template<typename T>
	static TypeInfo create() {
		static_assert(std::is_trivially_copyable<T>::value);
		static_assert(std::is_trivially_constructible<T>::value);
		return TypeInfo(std::type_index(typeid(T)), sizeof(T));
	}

	std::size_t getElementSize() { return elementSize; }
	operator std::type_index() { return typeIndex; }

private:
	TypeInfo(std::type_index typeIndex, std::size_t elementSize) : typeIndex(typeIndex), elementSize(elementSize) {}

private:
	std::type_index typeIndex;
	std::size_t elementSize;
};


struct DeviceMemory
{
private:
	DeviceMemory(void* ptr, count_t elemCount, TypeInfo valueType) : ptr(ptr), elemCount(elemCount), valueType(valueType) {}

private:
	void* ptr;
	count_t elemCount;
	TypeInfo valueType;

	friend struct ResourceManager;
};

using memory_t = std::shared_ptr<DeviceMemory>;