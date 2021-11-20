#pragma once

#define CHECK_CUDA(call)                                                                            \
do                                                                                                  \
{                                                                                                   \
	cudaError_t status = call;                                                                      \
	if (status != cudaSuccess) {                                                                    \
		auto message = fmt::format("cuda error: {} (code={})", cudaGetErrorString(status), status); \
		throw std::runtime_error(message);                                                          \
	}                                                                                               \
}                                                                                                   \
while(false)

#define CHECK_CUDA_NO_THROW(call)                                                           \
do                                                                                          \
{                                                                                           \
	cudaError_t status = call;                                                              \
	if (status != cudaSuccess) {                                                            \
		fmt::print(stderr, "cuda error: {} (code={})", cudaGetErrorString(status), status); \
		std::exit(EXIT_FAILURE);                                                            \
	}                                                                                       \
}                                                                                           \
while(false)
