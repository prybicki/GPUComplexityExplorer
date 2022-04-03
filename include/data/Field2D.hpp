#pragma once

#include <data/PropertySet.hpp>

struct Field2D
{
	Field2D(NCube2f area, Vec2i resolution, PropertySet properties) : _area(area), _resolution(resolution), _property(*properties.begin())
	{
		if (properties.getElementCount() != 1) {
			TODO("implement multi-layer field 2D");
		}
		Type type = _property.type();
		_current = mm.memoryAllocate(type, resolution.product());
		_previous = mm.memoryAllocate(type, resolution.product());
	}

	NCube2f getArea() { return _area; }
	NCube2i getCells(NCube2f area) {
		// TODO here
		// NCube functions
	}

private:
	NCube2f _area;
	Vec2i _resolution;
	Property _property;
	dev_mem_t _current;
	dev_mem_t _previous;
};