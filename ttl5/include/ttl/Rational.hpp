#pragma once

#include "Index.hpp"
#include <cassert>
#include <numeric>
#include <ostream>

namespace ttl {
struct Rational {
  std::ptrdiff_t p = 0;
  std::ptrdiff_t q = 1;

  constexpr Rational() = default;
  constexpr Rational(std::ptrdiff_t p) : p(p) {}
  constexpr Rational(std::ptrdiff_t p, std::ptrdiff_t q) : p(p), q(q) {
    assert(q != 0);
    auto d = std::gcd(p, q);
    p /= d;
    q /= d;
  }

  constexpr Rational inverse() const {
    return { q, p };
  }

  constexpr friend Rational operator+(Rational a) {
    return a;
  }

  constexpr friend Rational operator-(Rational a) {
    return { -a.p, a.q };
  }

  constexpr friend Rational operator+(Rational a, Rational b) {
    auto d = std::gcd(a.q, b.q);
    auto l = (a.q / d);
    auto r = (b.q / d);
    return { a.p * r + b.p * l, l * r };
  }

  constexpr friend Rational operator-(Rational a, Rational b) {
    auto d = std::gcd(a.q, b.q);
    auto l = (a.q / d);
    auto r = (b.q / d);
    return { a.p * r - b.p * l, l * r };
  }

  constexpr friend Rational operator*(Rational a, Rational b) {
    auto l = std::gcd(a.p, b.q);
    auto r = std::gcd(b.p, a.q);
    return { (a.p / l) * (b.p / r),
             (a.q / r) * (b.q / l) };
  }

  constexpr friend Rational operator/(Rational a, Rational b) {
    return a * b.inverse();
  }

  friend std::ostream& operator<<(std::ostream& os, Rational a) {
    return os << a.p << "/" << a.q;
  }
};
}
