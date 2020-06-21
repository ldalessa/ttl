#pragma once

#include "apply.hpp"
#include <cassert>
#include <concepts>
#include <variant>

namespace ttl::mp {

template <typename>
constexpr inline bool is_variant_v = false;

template <typename... Ts>
constexpr inline bool is_variant_v<std::variant<Ts...>> = true;

template <typename T>
concept Variant = is_variant_v<std::remove_cvref_t<T>>;

// see https://en.cppreference.com/w/cpp/utility/variant/visit
template <typename... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

// explicit deduction guide (not needed as of C++20)
template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename T, typename V>
concept VariantType = Variant<V> && requires(T, V v) { std::get<T>(v); };

template <Variant V, VariantType<V> T>
constexpr int offset_of(T&& t) {
  V v {t};                                      // <-- assume copy constructor
  const char* a = reinterpret_cast<const char*>(std::addressof(std::get<std::remove_cvref_t<T>>(v)));
  const char* b = reinterpret_cast<const char*>(std::addressof(v));
  return a - b;
}

template <Variant V, VariantType<V> T>
constexpr int index_of(const V v[], T&& t) {
  const char* a = reinterpret_cast<const char*>(std::addressof(t));
  const char* b = reinterpret_cast<const char*>(std::addressof(v[0]));
  const char* c = reinterpret_cast<const char*>(std::addressof(v[1]));

  int o = offset_of<V>(t);                      // offset of T within V [UB]
  int n = a - b - o;                            // bytes between t and v
  int m = c - b;                                // size of V within array
  assert(0 <= n);
  assert(n % m == 0);
  return n / m;                                 // index of t within v
}
}
