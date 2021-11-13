#include <kernels.hpp>

#include <cstdio>

__global__ void kHelloWorld(int value)
{
	printf("[%d][%d] Hello world x%d!\n", blockIdx.x, threadIdx.x, value);
}

