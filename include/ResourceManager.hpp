#pragma once

#include <fmt/format.h>

struct ResourceManager
{
	static ResourceManager& instance()
	{
		static ResourceManager instance;
		return instance;
	}

	template<typename Kernel, typename... Args>
	void runSync1D(int totalThreadCount, int blockThreadCount, Kernel&& kernel, Args... args) {
		void* kernelPtr = reinterpret_cast<void*>(&kernel);
		std::vector<void*> argsVec = intoPointers(args...);
		dim3 gridDim = {1U + totalThreadCount / blockThreadCount};
		dim3 blockDim = {static_cast<unsigned>(blockThreadCount), 1, 1};

		CHECK_CUDA(cudaLaunchKernel(kernelPtr, gridDim, blockDim, argsVec.data(), 0, 0));
		CHECK_CUDA(cudaStreamSynchronize(nullptr));
	}

private:
	ResourceManager() {}

	template<typename T, typename... Ts>
	std::vector<void*> intoPointers(T& arg, Ts&... args)
	{
		if constexpr (sizeof...(args) == 0) {
			return {&arg};
		}
		else {
			std::vector<void*> head = {&arg};
			std::vector<void*> tail = intoPointers(args...);
			head.insert(head.end(), tail.begin(), tail.end());
			return head;
		}
	}
};

ResourceManager& cm = ResourceManager::instance();
