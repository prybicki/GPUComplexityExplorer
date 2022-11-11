#pragma once


struct Type
{
	template<typename T>
	static Type create() {
		static_assert(std::is_trivially_copyable<T>::value);
		static_assert(std::is_trivially_constructible<T>::value);
		return {std::type_index(typeid(T)), sizeof(T)};
	}

	std::size_t sizeOf() { return elementSize; }

private:
	Type(std::type_index typeIndex, std::size_t elementSize) : typeIndex(typeIndex), elementSize(elementSize) {}

private:
	std::type_index typeIndex;
	std::size_t elementSize;
};

static Type f32 = Type::create<float>();
static Type f64 = Type::create<float>();
static Type i32 = Type::create<int32_t>();