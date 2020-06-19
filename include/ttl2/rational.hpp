#pragma once

#include "index.hpp"
#include <cassert>
#include <concepts>
#include <numeric>

namespace ttl {
struct rational {
  std::ptrdiff_t p = 0;
  std::ptrdiff_t q = 1;

  constexpr rational() = default;
  constexpr rational(std::ptrdiff_t p) : p(p) {}
  constexpr rational(std::ptrdiff_t p, std::ptrdiff_t q) : p(p), q(q) {
    assert(q != 0);
    auto d = std::gcd(p, q);
    p /= d;
    q /= d;
  }

  constexpr rational inverse() const {
    return { q, p };
  }

  friend constexpr rational operator+(rational a) {
    return a;
  }

  friend constexpr rational operator-(rational a) {
    return { -a.p, a.q };
  }

  friend constexpr rational operator+(rational a, rational b) {
    auto d = std::gcd(a.q, b.q);
    auto l = (a.q / d);
    auto r = (b.q / d);
    return { a.p * r + b.p * l, l * r };
  }

  friend constexpr rational operator-(rational a, rational b) {
    auto d = std::gcd(a.q, b.q);
    auto l = (a.q / d);
    auto r = (b.q / d);
    return { a.p * r - b.p * l, l * r };
  }

  friend constexpr rational operator*(rational a, rational b) {
    auto l = std::gcd(a.p, b.q);
    auto r = std::gcd(b.p, a.q);
    return { (a.p / l) * (b.p / r),
             (a.q / r) * (b.q / l) };
  }

  friend constexpr rational operator/(rational a, rational b) {
    return a * b.inverse();
  }
};

template <typename T>
concept Rational = std::same_as<rational, std::remove_cvref_t<T>>;
}
