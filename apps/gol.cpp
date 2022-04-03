#include <graphics/Visualizer.hpp>
#include <curand.h>
#include <unistd.h>
#include <core/ResourceManager.hpp>

#include <compute/kernels.hpp>

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)


int main(int argc, char** argv)
{
	Visualizer v(argc, argv);
	v.cameraSetCenter(1.5 * 4096, 4096);
	v.cameraSetMinRange(2 * 4096);
	const int sizeX = 4092;
	const int sizeY = 4092;

	const int megaX = 3;
	const int megaY = 2;

	cudaError_t status;
	void* curr[megaY][megaX];
	for (int y = 0; y < megaY; ++y) {
		for (int x = 0; x < megaX; ++x) {
			status = cudaMalloc(&curr[y][x], sizeX * sizeY);
			if (status != cudaSuccess) {
				std::terminate();
			}
		}
	}

	void* next[megaY][megaX];
	for (int y = 0; y < megaY; ++y) {
		for (int x = 0; x < megaX; ++x) {
			status = cudaMalloc(&next[y][x], sizeX * sizeY);
			if (status != cudaSuccess) {
				std::terminate();
			}
		}
	}

	for (int y = 0; y < megaY; ++y) {
		for (int x = 0; x < megaX; ++x) {
			status = cudaMemset(curr[y][x], 64, sizeX * sizeY);
			if (status != cudaSuccess) {
				std::terminate();
			}
		}
	}

	for (int y = 0; y < megaY; ++y) {
		for (int x = 0; x < megaX; ++x) {
			status = cudaMemset(next[y][x], 64, sizeX * sizeY);
			if (status != cudaSuccess) {
				std::terminate();
			}
		}
	}

	color_t* colorsTemp = nullptr;
	CHECK_CUDA(cudaMalloc(&colorsTemp, sizeof(color_t) * sizeX * sizeY));
	// float (*sNormalizeU8)(uint8_t) = nullptr;
	// CHECK_CUDA(cudaMemcpyFromSymbol(&sNormalizeU8, dPtrNormalizeU8, sizeof(sNormalizeU8)));


	curandGenerator_t gen;
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));


	for (int y = 0; y < megaY; ++y) {
		for (int x = 0; x < megaX; ++x) {
			CURAND_CALL(curandGenerate(gen, (unsigned int*) curr[y][x], sizeX * sizeY / 4));
			rm.run({sizeX * sizeY}, kSplit, curr[y][x]);
		}
	}
	bool play = true;
	int sleepMs = 0;
	v.setUserGUI([&](){
		ImGui::Begin("Control", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Checkbox("Play", &play);
		ImGui::SliderInt("Sleep", &sleepMs, 0, 1000);
		ImGui::End();
	});

	bool shouldContinue = true;
	while(shouldContinue) {
		for (int y = 0; y < megaY; ++y) {
			for (int x = 0; x < megaX; ++x) {
				rm.run({sizeX * sizeY}, kTmpColorizeCustomU8, sizeX * sizeY, reinterpret_cast<uint8_t*>(curr[y][x]), colorsTemp);
				v.renderTexture(x * (sizeX + 16), y * (sizeY+16), sizeX, sizeY, colorsTemp);
			}
		}
		if (play) {
			for (int y = 0; y < megaY; ++y) {
				for (int x = 0; x < megaX; ++x) {
					rm.run({sizeX * sizeY}, kGameOfLife, curr[y][x], next[y][x], sizeX, sizeY);
					std::swap(curr[y][x], next[y][x]);
				}
			}
			usleep(1000 * sleepMs);
		}
		v.redraw();
		shouldContinue = v.mainLoopIteration();
	}
}