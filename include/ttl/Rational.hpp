#pragma once

#include <fmt/format.h>
#include <cassert>
#include <numeric>
#include <string>
#include <utility>

namespace ttl {
struct Rational
{
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

  constexpr friend bool operator==(const Rational& a, const Rational& b) {
    return a.p == b.p && a.q == b.q;
  }

  std::string to_string() const {
    if (q != 1) {
      return std::to_string(p).append("/").append(std::to_string(q));
    }
    else {
      return std::to_string(p);
    }
  }
};
}

template <>
struct fmt::formatter<ttl::Rational> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const ttl::Rational& q, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", q.to_string());
  }
};
