#pragma once

#include <fmt/format.h>

#include <ResourcePrimitives.hpp>
#include <vector>

struct ResourceManager
{
	static ResourceManager& instance();

	template<typename Kernel, typename... KernelArgs>
	void run(ThreadsLayout threads, Kernel kernel, KernelArgs... kernelArgs)	{ run(threads, reinterpret_cast<void*>(kernel), std::vector<void*>{&kernelArgs...}.data());	}

	// TODO: type-check! Currently passing double instead of float -> UB
	template<typename T>
	memory_t memoryAllocate(count_t elements) { return memoryAllocate(TypeInfo::create<T>(), elements); }

private:
	ResourceManager() = default;

	void run(ThreadsLayout threads, void* kernel, void** args);

	memory_t memoryAllocate(TypeInfo valueType, count_t elements);
};

extern ResourceManager& rm;
