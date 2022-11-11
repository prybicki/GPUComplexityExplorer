#pragma once

#include <set>
#include <vector>
#include <stdexcept>
#include <string_view>

#include <data/Type.hpp>
#include <macros/iteration.hpp>
#include <data/Property.hpp>

struct PropertySet
{
	PropertySet(std::initializer_list<Property> properties)
	{
		std::set<std::string_view> names;
		for (auto&& property : properties) {
			auto&& [_, nameUnique] = names.insert(property.name());
			if (!nameUnique) {
				throw std::invalid_argument(fmt::format("duplicate property name: {}", property.name()));
			}
		}
		if (names.empty()) {
			throw std::invalid_argument(fmt::format("PropertySet cannot be empty"));
		}
		_properties = properties;
	}

	FORWARD_ITERATION(_properties)
	count_t getElementCount() { return _properties.size(); }

private:
	std::vector<Property> _properties;
};