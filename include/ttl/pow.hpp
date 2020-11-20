#pragma once

#include <concepts>

namespace ttl {
template <std::integral T>
constexpr T pow(T x, T y) {
  assert(y >= 0);
  T out = 1;
  for (T i = 0; i < y; ++i) {             // slow, buy hey, it's constexpr :-)
    out *= x;
  }
  return out;
}
}
