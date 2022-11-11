#pragma once

#include <fmt/format.h>

#include <core/ResourcePrimitives.hpp>
#include <vector>

struct MemoryManager
{
	static MemoryManager& instance();

	template<typename Kernel, typename... KernelArgs>
	void run(ThreadsLayout threads, Kernel kernel, KernelArgs... kernelArgs)	{ runLL(threads, reinterpret_cast<void*>(kernel), std::vector<void*>{&kernelArgs...}.data());	}

	template<typename T> // TODO: type-check! Currently passing double instead of float -> UB
	dev_mem_t memoryAllocate(count_t elements) { return memoryAllocate(Type::create<T>(), elements); }
	dev_mem_t memoryAllocate(Type valueType, count_t elements);

	void memoryFree(const DeviceMemory& memory);

private:
	MemoryManager() = default;

	void runLL(ThreadsLayout threads, void* kernel, void** args);

};

#define mm MemoryManager::instance()
