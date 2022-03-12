#pragma once


struct Type
{
	// TODO: Should Type has default value?
	template<typename T>
	static Type create() {
		static_assert(std::is_trivially_copyable<T>::value);
		static_assert(std::is_trivially_constructible<T>::value);
		return Type(std::type_index(typeid(T)), sizeof(T));
	}

	std::size_t sizeOf() { return elementSize; }

private:
	Type(std::type_index typeIndex, std::size_t elementSize) : typeIndex(typeIndex), elementSize(elementSize) {}

private:
	std::type_index typeIndex;
	std::size_t elementSize;
};