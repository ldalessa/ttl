#pragma once

#include <cassert>
#include <numeric>
#include <utility>
#include <fmt/core.h>

namespace ttl
{
  struct Rational
  {
    std::ptrdiff_t p = 0;
    std::ptrdiff_t q = 1;

    constexpr Rational() = default;

    constexpr Rational(std::ptrdiff_t p)
        : p(p)
    {
    }

    constexpr Rational(std::ptrdiff_t p, std::ptrdiff_t q)
        : p(p)
        , q(q)
    {
      assert(q != 0);
      auto d = std::gcd(p, q);
      p /= d;
      q /= d;
    }

    constexpr auto inverse() const -> Rational
    {
      return { q, p };
    }

    constexpr friend auto operator+(Rational const& a) -> Rational
    {
      return a;
    }

    constexpr friend auto operator-(Rational const& a) -> Rational
    {
      return { -a.p, a.q };
    }

    constexpr friend auto operator+(Rational const& a, Rational const& b)
      -> Rational
    {
      auto d = std::gcd(a.q, b.q);
      auto l = (a.q / d);
      auto r = (b.q / d);
      return { a.p * r + b.p * l, l * r };
    }

    constexpr auto operator+=(Rational const& b) -> Rational&
    {
      return *this = *this + b;
    }

    constexpr friend auto operator-(Rational const& a, Rational const& b)
      -> Rational
    {
      auto d = std::gcd(a.q, b.q);
      auto l = (a.q / d);
      auto r = (b.q / d);
      return { a.p * r - b.p * l, l * r };
    }

    constexpr auto operator-=(Rational const& b) -> Rational&
    {
      return *this = *this - b;
    }

    constexpr friend auto operator*(Rational const& a, Rational const& b)
      -> Rational
    {
      auto l = std::gcd(a.p, b.q);
      auto r = std::gcd(b.p, a.q);
      return {
        (a.p / l) * (b.p / r),
        (a.q / r) * (b.q / l)
      };
    }

    constexpr friend auto operator*=(Rational& a, Rational const& b)
      -> Rational&
    {
      return a = a * b;
    }

    constexpr friend auto operator/(Rational const& a, Rational const& b)
      -> Rational
    {
      return a * b.inverse();
    }

    constexpr friend bool operator==(Rational const&, Rational const&) = default;

    template <class T>
    constexpr friend auto as(Rational const& a) -> T
    {
      return T(a.p) / T(a.q);
    }

    constexpr friend auto pow(Rational const& a, Rational const& b) -> Rational
    {
      assert(b.q == 1);
      Rational c(1);
      for (int i = 0; i < b.p; ++i) {
        c *= a;
      }
      return c;
    }
  };

  namespace literals
  {
    constexpr Rational operator "" _q(unsigned long long n)
    {
      return Rational(n);
    }
  }
}

template <>
struct fmt::formatter<ttl::Rational> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::Rational& q, FormatContext& ctx) {
    if (q.q != 1) {
      return format_to(ctx.out(), "{}/{}", q.p, q.q);
    }
    else {
      return format_to(ctx.out(), "{}", q.p);
    }
  }
};
