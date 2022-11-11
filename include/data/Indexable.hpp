#pragma once

#include <concepts>
#include <types/count_t.hpp>

// Helper to get argument types from a method.

template<typename T>
struct FunctionTraits;

template<typename C, typename R, typename ...Args>
struct FunctionTraits<R(C::*)(Args...)>
{
	template <size_t i>
	struct Arg { typedef typename std::tuple_element<i, std::tuple<Args...>>::type type; };
};

// Following concepts could be probably expressed more generally, but I'm too scared to try.

template<typename T, typename R, typename I=int>
concept Indexable1D = requires(T t, I index)
{
	{ typename FunctionTraits<decltype(&T::operator[])>::template Arg<0>::type() } -> std::same_as<I>;
	{ t[index] } -> std::same_as<R&>;
	{ t.dims() } -> std::same_as<Vec1c>;
};

template<typename T, typename R, typename I2=int, typename I1=int>
concept Indexable2D = requires(T t, I2 index)
{
	{ typename FunctionTraits<decltype(&T::operator[])>::template Arg<0>::type() } -> std::same_as<I2>;
	{ t[index] } -> Indexable1D<R, I1>;
	{ t.dims() } -> std::same_as<Vec2c>;
};

template<typename T, typename R, typename I3=int, typename I2=int, typename I1=int>
concept Indexable3D = requires(T t, I2 index)
{
	{ typename FunctionTraits<decltype(&T::operator[])>::template Arg<0>::type() } -> std::same_as<I3>;
	{ t[index] } -> Indexable2D<R, I2, I1>;
	{ t.dims() } -> std::same_as<Vec3c>;
};
