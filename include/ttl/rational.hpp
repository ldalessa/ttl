#pragma once

#include "index.hpp"
#include <ratio>
#include <string>

namespace ttl {
template <std::ptrdiff_t a, std::ptrdiff_t b>
struct rational {
  using q = std::ratio<a, b>;
  constexpr rational() = default;
  constexpr rational(q) {}
};

inline constexpr rational< 0, 1>    zero = {};
inline constexpr rational< 1, 1>     one = {};
inline constexpr rational<-1, 1> neg_one = {};

template <typename>
inline constexpr bool is_rational_v = false;

template <std::ptrdiff_t a, std::ptrdiff_t b>
inline constexpr bool is_rational_v<rational<a, b>> = true;

template <typename T>
concept Rational = is_rational_v<T>;

template <Rational A, Rational B>
constexpr bool operator==(A, B) {
  return std::ratio_equal<typename A::q, typename B::q>();
}

template <typename T>
concept Zero = Rational<T> && (std::remove_cvref<T>() == zero);

template <typename T>
concept One = Rational<T> && (std::remove_cvref<T>() == one);

template <Rational A, Rational B>
constexpr auto operator+(A, B) {
  return rational(std::ratio_add<typename A::q, typename B::q>());
}

template <Rational A, Rational B>
constexpr auto operator-(A, B) {
  return rational(std::ratio_subtract<typename A::q, typename B::q>());
}

template <Rational A, Rational B>
constexpr auto operator*(A, B) {
  return rational(std::ratio_multiply<typename A::q, typename B::q>());
}

template <Rational A, Rational B>
constexpr auto operator/(A, B) {
  return rational(std::ratio_divide<typename A::q, typename B::q>());
}

template <Rational A>
constexpr auto operator-(A a) {
  constexpr rational<-1, 1> neg_one;
  return neg_one * a;
}

template <std::ptrdiff_t a, std::ptrdiff_t b>
constexpr auto name(rational<a, b>) {
  return std::to_string(a) + "/" + std::to_string(b);
}

template <Rational A>
constexpr int order(A) {
  return 0;
}

template <Rational T>
constexpr index<0> outer(T) {
  return {};
}

template <Rational T>
constexpr decltype(auto) rewrite(T v, index<0>) {
  return v;
}
}
