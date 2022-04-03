#pragma once

#include <data/Type.hpp>

struct Property
{
	Property(Type type, const std::string name="", bool doubleBuffered=false, count_t history=0) : type(type) {
		// TODO
	}

	const std::string_view getName() const { return name; }

private:
	Type type;
	std::string name;
};

struct PropertySet
{
	PropertySet(std::initializer_list<Property> properties)
	{
		std::set<std::string_view> names;
		for (auto&& property : properties) {
			auto&& [_, nameUnique] = names.insert(property.getName());
			if (!nameUnique) {
				throw std::invalid_argument(fmt::format("duplicate property name: {}", property.getName()));
			}
		}
	}
private:

};

// struct BoundProperty : public Property
// {
// 	std::shared_ptr<ICompoundData> parent;
// };

// TODO: after proper benchmarks (copy, compute, map), decide what memory kind to use (linear, pitched, surface)
struct Field2D
{
	Field2D(NCube2f area, Vec2i resolution, PropertySet properties) : area(area), resolution(resolution)
	{

	}

	NCube2f getArea() { return area; }

private:
	NCube2f area;
	Vec2i resolution;
};