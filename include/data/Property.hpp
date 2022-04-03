#pragma once

#include <data/Type.hpp>

struct Property
{
	Property(Type type, const std::string name="") : _type(type), _name(name) { }
	const std::string_view name() const { return _name; }
	const Type& type() const { return _type; }
	auto operator<=>(const Property& other) { return name() <=> other.name(); }
private:
	Type _type;
	std::string _name;
};
