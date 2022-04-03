#pragma once

#include <math/Vector.hpp>
#include <macros/todo.hpp>
#include <macros/cuda.hpp>

template<count_t dim, typename T>
struct NCube
{
	using VecT = Vector<dim, T>;
	using NCubeT = NCube<dim, T>;

	static NCubeT centeredAt(VecT center, VecT dims)
	{ return NCubeT(center - dims.half(), center + dims.half()); }

	static NCubeT spannedBetween(VecT _min, VecT _max)
	{ return NCubeT(_min, _max); }

	HD VecT min() const { return _min; }
	HD VecT max() const { return _max; }
	HD VecT span() const {return (max() - min()); }

	HD VecT center() const requires std::is_floating_point_v<T>
	{ return min() + (max() - min()).half(); }

	HD NCubeT scaled(VecT scale) const requires std::is_floating_point_v<T>
	{ return centeredAt(center(), scale * span()); }

	HD NCubeT shrinkedBy(VecT amount) const requires std::is_floating_point_v<T>
	{return centeredAt(center(), span() - amount); }

	HD VecT positionOf(Vector<dim, float> positionFracts) const requires std::is_floating_point_v<T>
	{ return min() + positionFracts * (max() - min()); }

	HD NCubeT placedIn(NCubeT enclosing, Vector<dim, float> positionFracts) const requires std::is_floating_point_v<T>
	{ return NCubeT::centeredAt(enclosing.shrinkedBy(span()).positionOf(positionFracts), span()); }


private:
	NCube(const VecT& min, const VecT& max) : _min(min), _max(max) {}
	VecT _min;
	VecT _max;
};

template<int dim, typename T>
struct fmt::formatter<NCube<dim, T>>
{
	template<typename ParseContext>
	constexpr auto parse(ParseContext& ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(NCube<dim, T> const& nc, FormatContext& ctx) {
		return fmt::format_to(ctx.out(), "[{} -> {}]", nc.min(), nc.max());
	}
};

using NCube2f = NCube<2, float>;
using NCube2i = NCube<2, int>;

static_assert(std::is_trivially_copyable<NCube2f>::value);
static_assert(std::is_trivially_copyable<NCube2i>::value);
