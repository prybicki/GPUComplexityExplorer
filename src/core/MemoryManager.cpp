#include <core/MemoryManager.hpp>

#include <macros/todo.hpp>
#include <macros/cuda.hpp>

#include <cuda_runtime_api.h>
#include <core/ResourcePrimitives.hpp>

void MemoryManager::run(ThreadsLayout threads, void *kernel, void **args)
{
	CHECK_CUDA(cudaLaunchKernel(kernel, threads.gridDim, threads.blockDim, args, 0, 0));
	CHECK_CUDA(cudaStreamSynchronize(nullptr));
}

MemoryManager &MemoryManager::instance()
{
	static MemoryManager instance;
	return instance;
}

ThreadsLayout::ThreadsLayout(count_t launchDim, count_t blockDim)
: gridDim(1U + launchDim / blockDim)
, blockDim(static_cast<unsigned>(blockDim))
{}

ThreadsLayout::ThreadsLayout(std::array<count_t, 2> launchDims, std::array<count_t, 2> blockDims)
: gridDim(1U + launchDims[0] / blockDims[0], 1U + launchDims[1] / blockDims[1])
, blockDim(static_cast<unsigned>(blockDims[0]), static_cast<unsigned>(blockDims[1]))
{

}

dev_mem_t MemoryManager::memoryAllocate(Type valueType, count_t elements)
{
	void* ptr = nullptr;
	CHECK_CUDA(cudaMalloc(&ptr, valueType.sizeOf() * elements));
	return std::shared_ptr<DeviceMemory>(new DeviceMemory(ptr, elements, valueType));
}

void MemoryManager::memoryFree(const DeviceMemory &memory)
{
	CHECK_CUDA(cudaFree(memory.ptr));
}

DeviceMemory::~DeviceMemory()
{
	mm.memoryFree(*this);
}
