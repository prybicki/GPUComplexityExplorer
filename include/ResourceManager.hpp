#pragma once

#include <fmt/format.h>

#include <ResourcePrimitives.hpp>

struct ResourceManager
{
	static ResourceManager& instance();

	template<typename Kernel, typename... KernelArgs>
	void run(ThreadsLayout threads, Kernel kernel, KernelArgs... kernelArgs)	{ run(threads, reinterpret_cast<void*>(kernel), (void*[]){&kernelArgs...});	}

private:
	ResourceManager() = default;

	void run(ThreadsLayout threads, void* kernel, void** args);
};

extern ResourceManager& rm;
