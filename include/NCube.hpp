#pragma once

#include <Vector.hpp>

template<count_t dim, typename T>
struct NCube
{
	Vector<dim, T> min;
	Vector<dim, T> max;
};

using NCube2f = NCube<2, float>;
using NCube2i = NCube<2, int>;

static_assert(std::is_trivially_copyable<NCube2f>::value);
