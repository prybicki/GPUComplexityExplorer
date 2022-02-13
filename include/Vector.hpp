#pragma once

#include <type_traits>
#include <numeric>
#include <array>

#include <fmt/format.h>

#include <internal/aliases.hpp>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif // __CUDACC__

template<count_t dim, typename T>
class Vector
{
	using V = Vector<dim, T>;
	T row[dim];

public:
	__host__ __device__
	Vector() {
		for (int i = 0; i < dim; ++i) {
			row[i] = static_cast<T>(0);
		}
	}

	__host__
	Vector(std::array<T, dim> scalar)
	{
		for (int i = 0; i < dim; ++i) {
			row[i] = scalar[i];
		}
	}

	template<typename... Args>
	__host__
	Vector(Args... args) : Vector(std::array<T, dim> {args...}) {}

	// Cast constructor
	template<typename U>
	Vector(const Vector<dim, U>& other) {
		for (count_t i = 0; i < dim; ++i) {
			row[i] = static_cast<T>(other.row[i]);
		}
	}

	template<int lowerDim>
	__host__ __device__
	Vector(const Vector<lowerDim, T>& other, T fillValue)
	{
		static_assert(lowerDim < dim);
		for (int i = 0; i < lowerDim; ++i) {
			row[i] = other[i];
		}
		for (int i = lowerDim; i < dim; ++i) {
			row[i] = fillValue;
		}
	}

	__host__ __device__
	V& operator+=(const V& rhs)
	{
		for (count_t i = 0; i < dim; ++i) {
			row[i] += rhs[i];
		}
		return *this;
	}

	__host__ __device__
	friend V operator+(V lhs, const V& rhs)
	{
		lhs += rhs;
		return lhs;
	}

	__host__ __device__
	V& operator-=(const V& rhs)
	{
		for (count_t i = 0; i < dim; ++i) {
			row[i] -= rhs[i];
		}
		return *this;
	}

	__host__ __device__
	friend V operator-(V lhs, const V& rhs)
	{
		lhs -= rhs;
		return lhs;
	}

	__host__ __device__
	V& operator*=(T rhs)
	{
		for (count_t i = 0; i < dim; ++i) {
			row[i] *= rhs;
		}
		return *this;
	}

	__host__ __device__
	friend V operator*(T lhs, V rhs)
	{
		rhs *= lhs;
		return rhs;
	}

	__host__ __device__
	V& operator/=(T rhs)
	{
		for (count_t i = 0; i < dim; ++i) {
			row[i] /= rhs;
		}
		return *this;
	}

	__host__ __device__
	friend V operator/(V lhs, T rhs)
	{
		lhs /= rhs;
		return lhs;
	}

	__host__ __device__
	T& x() { static_assert(dim > 0); return row[0]; }

	__host__ __device__
	T& y() { static_assert(dim > 1); return row[1]; }

	__host__ __device__
	T& z() { static_assert(dim > 2); return row[1]; }

	__host__ __device__
	T& operator[](int i) { return row[i]; }

	__host__ __device__
	const T& operator[](int i) const { return row[i]; }

	__host__ __device__
	T length2() const {
		T sum = 0;
		for (count_t i = 0; i < dim; ++i) {
			sum += row[i] * row[i];
		}
		return sum;
	}

	__host__ __device__
	T length() const {
		return std::sqrt(length2());
	}

};

#ifndef __CUDACC__
template<int dim, typename T>
struct fmt::formatter<Vector<dim, T>>
{
	template<typename ParseContext>
	constexpr auto parse(ParseContext& ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(Vector<dim, T> const& vec, FormatContext& ctx) {

		fmt::format_to(ctx.out(), "(");
		for (auto&& i : vec.dims()) {
			fmt::format_to(ctx.out(), "{}", vec[i]);
			if (i < dim - 1) {
				fmt::format_to(ctx.out(), ", ");
			}
		}
		return fmt::format_to(ctx.out(), ")");
	}
};
#endif // __CUDACC__


//template class Vec<2, float>;
template<typename T>
using Vec2 = Vector<2, T>;
using Vec2f = Vector<2, float>;
using Vec3f = Vector<3, float>;
using Vec4f = Vector<4, float>;

static_assert(std::is_trivially_copyable<Vec2f>::value);
static_assert(std::is_trivially_copyable<Vec3f>::value);
static_assert(std::is_trivially_copyable<Vec4f>::value);
