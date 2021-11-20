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
