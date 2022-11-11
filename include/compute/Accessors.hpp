#pragma once

#include <macros/cuda.hpp>
#include <math/Vector.hpp>

template<typename T>
struct LinMemAcc1D
{
	HD LinMemAcc1D(T* data, Vec1c dims) : _data(data), _dims(dims) {}
	HD T& operator[](count_t i) { return _data[i]; }
	HD Vec1c dims() { return _dims; }
private:
	T* _data;
	Vec1c _dims;
};

template<typename T>
struct LinMemAcc2D
{
	HD LinMemAcc2D(T* data, Vec2c dims) : _data(data), _dims(dims) {}

	HD LinMemAcc1D<T> operator[](int i) { return LinMemAcc1D(_data + dims().x() * i, dims().x());}
	HD Vec2c dims() { return _dims; }
private:
	T* _data;
	Vec2c _dims;
};
