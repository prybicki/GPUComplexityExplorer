#include <compute/kernels.hpp>

#include <math/Vector.hpp>
#include <math/Matrix.hpp>

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
		transform.refColumn<2>() = Vec3f(position[i], 1.0f);
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

// __device__ float dNormalizeU8(uint8_t in)
// {
// 	return (static_cast<float>(in) / 255.0f);
// }

// __device__ float dIdentity(float in)
// {
// 	return in;
// }

// __device__ float (*dPtrNormalizeU8)(uint8_t) = dNormalizeU8;
//
// template<typename InputType>
// __global__ void kColorizeBlackToWhite(count_t count, InputType* in, color_t* out, float (*normalize)(InputType))
// {
// 	int tid = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (tid >= count) {
// 		return;
// 	}
// 	// TODO: test it (does it cover whole range?)
// 	uint8_t value = static_cast<uint8_t>(255.0f * normalize(in[tid]));
// 	out[tid].r = value;
// 	out[tid].g = value;
// 	out[tid].b = value;
// 	out[tid].a = 255;
// }
//
// template __global__ void kColorizeBlackToWhite<uint8_t>(count_t count, uint8_t* in, color_t* out, float (*normalize)(uint8_t));
// template __global__ void kColorizeBlackToWhite<float>(count_t count, float* in, color_t* out, float (*normalize)(float));

__global__ void kTmpColorizeCustomU8(count_t count, const uint8_t* in, color_t* out)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= count) {
		return;
	}
	bool even = (tid % 2) == 0;
	bool odd = !even;
	if (in[tid] < 128) {
		out[tid].r = (1+even) * in[tid];
		out[tid].b = (1+odd) * in[tid];
		out[tid].g = 0;
	}
	else {
		out[tid].r = 0;
		out[tid].g = in[tid];
		out[tid].b = in[tid];
	}
	out[tid].a = 255;
}

// __global__ void kTmpColorizeCustomF32(count_t count, const float* in, const float* inMin, const float* inMax,  color_t* out)
// {
// 	int tid = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (tid >= count) {
// 		return;
// 	}
// 	// float min = *inMin;
// 	// float max = *inMax;
// 	float min = 0.0f;
// 	float max = 1.0f;
// 	uint8_t value = static_cast<uint8_t>(255 * (in[tid] - min) / (max - min));
// 	out[tid].r = value;
// 	out[tid].g = 0;
// 	out[tid].b = 0;
// 	out[tid].a = 255;
// }

__global__ void kTmpColorizeCustomF32(count_t count, const float* in, const float* inMin, const float* inMax,  color_t* out)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= count) {
		return;
	}
	// float min = *inMin;
	// float max = *inMax;
	float min = 0.0f;
	float max = 1.0f;
	float fval = (in[tid] - min) / (max - min);

	if (fval < 0.2) {
		float progress = (fval - 0.0f) / 0.2f;
		uint8_t uval = static_cast<uint8_t>(255.0f * progress);
		out[tid].r = uval;
		out[tid].g = 0;
		out[tid].b = 0;
	}
	else if (fval < 0.4) {
		float progress = (fval - 0.2f) / 0.2f;
		uint8_t uval = static_cast<uint8_t>(255.0f * progress);
		out[tid].r = 255;
		out[tid].g = uval;
		out[tid].b = 0;
	}
	else if (fval < 0.6) {
		float progress = (fval - 0.4f) / 0.2f;
		uint8_t uval = static_cast<uint8_t>(255.0f * progress);
		out[tid].r = 255-uval;
		out[tid].g = 255;
		out[tid].b = 0;
	}
	else if (fval < 0.8) {
		float progress = (fval - 0.6f) / 0.2f;
		uint8_t uval = static_cast<uint8_t>(255.0f * progress);
		out[tid].r = 0;
		out[tid].g = 255;
		out[tid].b = uval;
	}
	else {
		float progress = (fval - 0.8f) / 0.2f;
		uint8_t uval = static_cast<uint8_t>(255.0f * progress);
		out[tid].r = 0;
		out[tid].g = 255-uval;
		out[tid].b = 255;
	}
	out[tid].a = 255;
}

__global__ void kTmpSetNCube(count_t width, count_t height, float* data, NCube2i rect, float value)
{
	int tX = threadIdx.x + blockIdx.x * blockDim.x;
	int tY = threadIdx.y + blockIdx.y * blockDim.y;
	bool inRange = (tX < width) && (tY < height);
	bool inNCube = (rect.min().x() <= tX) && (tX < rect.max().x())
	            && (rect.min().y() <= tY) && (tY < rect.max().y());
	if (!inRange || !inNCube) {
		return;
	}
	data[tX + tY * width] = value;
}

__global__ void kHeatTransfer(count_t width, count_t height, const float* curr, float* next, float coeff)
{
	int tX = threadIdx.x + blockIdx.x * blockDim.x;
	int tY = threadIdx.y + blockIdx.y * blockDim.y;
	bool inRange = (tX < width) && (tY < height);
	if (!inRange) {
		return;
	}
	bool isBorder = (tX == 0) || (tY == 0) || (tX == width - 1) || (tY == height - 1);
	if (isBorder) {
		next[tX + tY * width] = 0.0f;
		return;
	}

	next[tX + tY * width] = curr[tX + tY * width];
	next[tX + tY * width] += coeff * (curr[(tX+1) + (tY) * width] - curr[tX + tY * width]);
	next[tX + tY * width] += coeff * (curr[(tX-1) + (tY) * width] - curr[tX + tY * width]);
	next[tX + tY * width] += coeff * (curr[(tX) + (tY+1) * width] - curr[tX + tY * width]);
	next[tX + tY * width] += coeff * (curr[(tX) + (tY-1) * width] - curr[tX + tY * width]);
}