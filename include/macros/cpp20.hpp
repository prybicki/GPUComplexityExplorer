#pragma once

// This macro is useful for optional C++20 syntax, e.g. requires (...)
//
#ifdef __NVCC__
#define CPP20(expr)
#else
#define CPP20(expr) expr
#endif // __NVCC__