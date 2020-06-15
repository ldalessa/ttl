#pragma once

#include "index.hpp"
#include <concepts>
#include <string>
#include <type_traits>

namespace ttl
{
template <typename T>
concept Arithmetic =
 std::signed_integral<std::remove_cvref_t<T>> ||
 std::floating_point<std::remove_cvref_t<T>>;

template <Arithmetic T>
constexpr auto name(T v) {
  return std::to_string(v);
}

template <Arithmetic T>
constexpr int order(T) {
  return 0;
}

template <Arithmetic T>
constexpr decltype(auto) outer(T) {
  return index();
}

template <Arithmetic T, Index Is>
constexpr decltype(auto) rewrite(T v, Is is) {
  assert(size(is) == 0);
  assert(v != 0);
  return v;
}
}
