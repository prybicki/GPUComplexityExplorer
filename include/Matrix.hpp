#pragma once

#include <Vector.hpp>
#include <array>

// #ifndef __CUDACC__
// #define __host__
// #define __device__
// #endif // __CUDACC__

template<count_t columnCount, count_t rowCount, typename T>
class Matrix
{
	using M = Matrix<columnCount, rowCount, T>;
	using V = Vector<rowCount, T>;
	V column[columnCount];

public:
	struct IdentityInitT {};

	__host__ __device__
	Matrix()
	{
		for (count_t i = 0; i < columnCount; ++i) {
			column[i] = V(0.0f);
		}
	}

	__host__ __device__
	Matrix(IdentityInitT, T value)
	{
		count_t minDim = columnCount < rowCount ? columnCount : rowCount;
		for (count_t i = 0; i < minDim; ++i) {
			this->operator[](i)[i] = value;
		}
	}

	template<int i>
	__host__ __device__
	V& refColumn() { return column[i]; }

	__host__ __device__
	V& operator[](count_t i){ return column[i]; }
};

using Mat3x3f = Matrix<3, 3, float>;
static_assert(std::is_trivially_copyable<Mat3x3f>::value);
