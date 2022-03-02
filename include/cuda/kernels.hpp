#pragma once

#include <cuda_runtime.h>
#include <Vector.hpp>
#include <Matrix.hpp>
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