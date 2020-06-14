#pragma once

#include <concepts>
#include <string>

namespace ttl
{
template <typename T>
concept Arithmetic = std::signed_integral<T> || std::floating_point<T>;

template <Arithmetic T>
constexpr auto name(T v) {
  return std::to_string(v);
}

template <Arithmetic T>
constexpr int order(T) {
  return 0;
}

template <Arithmetic T>
constexpr index<0> outer(T) {
  return {};
}

template <Arithmetic T>
constexpr decltype(auto) rewrite(T v, index<0>) {
  return v;
}
}
