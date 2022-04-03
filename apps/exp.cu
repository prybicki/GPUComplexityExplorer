#include <cstddef>
#include <tuple>
#include <iostream>
#include <compute/kernels.hpp>
#include <compute/Accessors.hpp>

int main()
{
	int* ptr;
	cudaMalloc(&ptr, 4 * 4 * 4);
	kSetMap<<<1,1>>>(LinMemAcc2D<int>(ptr, {4, 4}));
	kPrintMap<<<1,1>>>(LinMemAcc2D<int>(ptr, {4, 4}));
	cudaDeviceSynchronize();
}
