#pragma once

#define TODO(what)                                                                      \
do {                                                                                    \
	throw std::logic_error(fmt::format("TODO: {} at {}:{}", what, __FILE__, __LINE__)); \
} while(0)
