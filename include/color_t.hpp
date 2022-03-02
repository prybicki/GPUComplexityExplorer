#pragma once

#include <cstdint>
#include <cmath>
#include <fmt/format.h>
#include <type_traits>

struct color_t {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;
};
static_assert(sizeof(uint32_t) == sizeof(color_t));
static_assert(std::is_trivially_copyable<color_t>::value);

template<>
struct fmt::formatter<color_t>
{
  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
      return ctx.begin();
  }

  template<typename FormatContext>
  auto format(color_t const& c, FormatContext& ctx) {
      return fmt::format_to(ctx.out(), "({}, {}, {}, {})", c.r, c.g, c.b, c.a);
  }
};
