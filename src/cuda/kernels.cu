#include <cuda/kernels.hpp>

#include <Vector.hpp>
#include <Matrix.hpp>

using count_t = int;

#define FOR(var, range)\
	for (count_t var = blockDim.x * blockIdx.x + threadIdx.x; \
	     (var) < (range);                                     \
	     (var) += gridDim.x * blockDim.x                      \
 )

__global__ void kHelloWorld(int value)
{
	printf("[%d][%d] Hello world x%d!\n", blockIdx.x, threadIdx.x, value);
}

__global__ void kPos2DToTransform3x3(count_t count, const Vec2f* position, const float* radius, Mat3x3f* outTransform)
{
	FOR(i, count) {
		Mat3x3f transform = {Mat3x3f::IdentityInitT{}, 1.0f};
		transform[0][0] = radius[i];
		transform[1][1] = radius[i];
		transform.refColumn<2>() = Vec3f(position[i], 1);
		outTransform[i] = transform;
	}
}

__global__ void kApplyVelocity(count_t count, float dt, const Vec2f* vel, Vec2f* pos)
{
	FOR(i, count) {
		pos[i] += dt * vel[i];
	}
}

__global__ void kGameOfLife(unsigned char* in, unsigned char* out, int width, int height)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int x = tid % width;
	int y = tid / width;
	if (x >= width || y >= height) {
		return;
	}

	unsigned char currValue;
	unsigned char count = 0;
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			if (!(0 <= x + dx && x + dx < width)) {
				continue;
			}
			if (!(0 <= y + dy && y + dy < height)) {
				continue;
			}
			if (dx == 0 && dy == 0) {
				currValue = in[y*width + x];
				continue;
			}
			count += (in[width * (y+dy) + (x+dx)] == 255) ? 1 : 0;
		}
	}
	unsigned char nextValue = currValue;
	if (currValue < 255 && count == 3) {
		nextValue = 255; // becomes alive
	}
	if (currValue == 255) {
		bool stayAlive = (count == 2 || count == 3);
		nextValue = stayAlive ? 255 : 100;
	}
	if (nextValue < 255 && nextValue > 0) {
		nextValue -= 2;
	}

	out[y * width + x] = nextValue;
}

// TODO: tmp workaround, remove me
__global__ void kSplit(unsigned char* data)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned char* v = &data[tid];
	*v = *v > 127 ? 255 : 0;
}