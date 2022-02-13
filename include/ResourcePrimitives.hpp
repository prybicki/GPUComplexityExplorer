#pragma once

#include <internal/aliases.hpp>
#include <array>

struct ThreadsLayout
{
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
