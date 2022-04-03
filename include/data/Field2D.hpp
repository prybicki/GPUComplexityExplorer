#pragma once

#include <data/PropertySet.hpp>
#include <data/Indexable.hpp>

struct Field2D
{
	Field2D(NCube2f area, Vec2c resolution, PropertySet properties) : _area(area), _resolution(resolution), _property(*properties.begin())
	{
		if (properties.getElementCount() != 1) {
			TODO("implement multi-layer field 2D");
		}
		Type type = _property.type();
		_current = mm.memoryAllocate(type, resolution.product());
		_previous = mm.memoryAllocate(type, resolution.product());
	}

	template<typename R, Indexable2D<R> Map2D>
	Map2D property(const std::string_view name)
	{
		return _current->accessor2D<R>(_resolution);
	}

	Vec2c dims() { return _resolution; }
	NCube2f area() { return _area; }
	NCube2c cellsIn(NCube2f area) {
		// TODO here
		// NCube functions
	}

private:
	NCube2f _area;
	Vec2c _resolution;
	Property _property;
	dev_mem_t _current;
	dev_mem_t _previous;
};