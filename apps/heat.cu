#include <Visualizer.hpp>
#include <ResourceManager.hpp>
#include <macros.hpp>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <cuda/kernels.hpp>

#include <data/data.hpp>

#include <unistd.h>

int main(int argc, char** argv)
{
	Visualizer v {argc, argv};
	const int sizeX = 1024;
	const int sizeY = 1024;
	float* fieldCurr = nullptr;
	float* fieldNext = nullptr;
	color_t* fieldCol = nullptr;

	v.setCameraCenter(sizeX / 2, sizeY / 2);
	v.setCameraMinRange(sizeY * 1.5f);

	CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&fieldCol), sizeof(color_t) * sizeX * sizeY));
	CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&fieldCurr), sizeof(float) * sizeX * sizeY));
	CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&fieldNext), sizeof(float) * sizeX * sizeY));
	CHECK_CUDA(cudaMemset(fieldCurr, 0, sizeof(float) * sizeX * sizeY));

	NCube2i rect1 = {.min={1 * sizeX/4, 1 * sizeY/4}, .max={1 * sizeX/4 + 1 * 42, 1 * sizeY/4 + 1 * 42}};
	NCube2i rect2 = {.min={2 * sizeX/4, 2 * sizeY/4}, .max={2 * sizeX/4 + 2 * 42, 2 * sizeY/4 + 2 * 42}};
	NCube2i rect3 = {.min={3 * sizeX/4, 3 * sizeY/4}, .max={3 * sizeX/4 + 3 * 42, 3 * sizeY/4 + 3 * 42}};

	NCube2i rect4 = {.min={1 * sizeX/4, 2 * sizeY/4}, .max={1 * sizeX/4 + 2 * (sizeX/16), 2 * sizeY/4 + 6 * (sizeY/16)}};
	NCube2i rect5 = {.min={2 * sizeX/4, 1 * sizeY/4}, .max={2 * sizeX/4 + 6 * (sizeX/16), 1 * sizeY/4 + 2 * (sizeY/16)}};
	rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect1, 1.0f);
	rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect2, 0.9f);
	rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect3, 0.8f);

	rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect4, 0.8f);
	rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect5, 0.8f);

	while (v.mainLoopIteration())
	{
		auto begin = thrust::device_ptr<float>(fieldCurr);
		auto end = thrust::device_ptr<float>(fieldCurr + sizeX * sizeY);
		thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>> minMaxPtrs = thrust::minmax_element(begin, end);
		rm.run({sizeX * sizeY}, kTmpColorizeCustomF32, sizeX * sizeY, fieldCurr, minMaxPtrs.first.get(), minMaxPtrs.second.get(), fieldCol);
		v.renderTexture(0, 0, sizeX, sizeY, fieldCol);
		v.redraw();

		for (int i = 0; i < 100; i++) {
			rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kHeatTransfer, sizeX, sizeY, fieldCurr, fieldNext, 0.01f);
			std::swap(fieldCurr, fieldNext);

		}
	}
}
