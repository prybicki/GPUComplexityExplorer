#include <ResourceManager.hpp>

#include <macros.hpp>

#include <cuda_runtime_api.h>
#include <ResourcePrimitives.hpp>

ResourceManager& rm = ResourceManager::instance();

void ResourceManager::run(ThreadsLayout threads, void *kernel, void **args)
{
	CHECK_CUDA(cudaLaunchKernel(kernel, threads.gridDim, threads.blockDim, args, 0, 0));
	CHECK_CUDA(cudaStreamSynchronize(nullptr));
}

ResourceManager &ResourceManager::instance()
{
	static ResourceManager instance;
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

memory_t ResourceManager::memoryAllocate(Type valueType, count_t elements)
{
	void* ptr = nullptr;
	CHECK_CUDA(cudaMalloc(&ptr, valueType.sizeOf() * elements));
	return std::shared_ptr<DeviceMemory>(new DeviceMemory(ptr, elements, valueType));
}