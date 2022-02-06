#pragma once

#include <cuda_runtime.h>
#include <Vector.hpp>
#include <Matrix.hpp>

using count_t = int;

__global__ void kHelloWorld(int value);
__global__ void kPos2DToTransform3x3(count_t count, const Vec2f* position, const float* radius, Mat3x3f* outTransform);
__global__ void kApplyVelocity(count_t count, float dt, const Vec2f* vel, Vec2f* pos);
__global__ void kGameOfLife(unsigned char* in, unsigned char* out, int width, int height);
__global__ void kSplit(unsigned char* data);