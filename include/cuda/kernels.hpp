#pragma once

#include <cuda_runtime.h>
#include <math/Vector.hpp>
#include <math/Matrix.hpp>
#include <color_t.hpp>

using count_t = int;

__global__ void kHelloWorld(int value);
__global__ void kPos2DToTransform3x3(count_t count, const Vec2f* position, const float* radius, Mat3x3f* outTransform);
__global__ void kApplyVelocity(count_t count, float dt, const Vec2f* vel, Vec2f* pos);
__global__ void kGameOfLife(unsigned char* in, unsigned char* out, int width, int height);
__global__ void kSplit(unsigned char* data);


// TODO: normalize should be a form of partial function, e.g. a functor.
// TODO: add alpha

// TODO: making colorization even somewhat generic is a bit blocked by
// TODO: the fact that taking __device__ function address from __host__ is illegal
// TODO: this should be researched

// template<typename InputType>
// __global__ void kColorizeBlackToWhite(count_t count, InputType* in, color_t* out, float (*normalize)(InputType));
//
// extern template
// __global__ void kColorizeBlackToWhite<uint8_t>(count_t count, uint8_t* in, color_t* out, float (*normalize)(uint8_t));
//
// extern template
// __global__ void kColorizeBlackToWhite<float>(count_t count, float* in, color_t* out, float (*normalize)(float));

// Temporary manual solution:
__global__ void kTmpColorizeCustomU8(count_t count, const uint8_t* in, color_t* out);

__global__ void kTmpColorizeCustomF32(count_t count, const float* in, const float* inMin, const float*inMax,  color_t* out);

#include <math/NCube.hpp>

// TODO: Can gridDim represent actual data size?
__global__ void kTmpSetNCube(count_t width, count_t height, float* data, NCube2i rect, float value);

__global__ void kHeatTransfer(count_t width, count_t height, const float* curr, float* next, float coeff);