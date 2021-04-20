#pragma once

#include <cmath>

namespace ttl
{
  constexpr uint64_t sign(double d)
  {
    return std::signbit(d);
  }

  constexpr uint64_t exponent(double d)
  {
    return std::ilogb(d) + 1023;
  }

  constexpr uint64_t mantissa(double d)
  {
    // clear this 53rd bit (the significand's hidden bit has been exposed)
    constexpr uint64_t mask = (uint64_t(1) << 52) - 1;
    uint64_t i = std::abs(std::scalbn(d, 52 - std::ilogb(d)));
    return i & mask;
  }

  constexpr uint64_t pack_fp(double d)
  {
    if (d == 0.0) return 0;
    if (d == -0.0) return uint64_t(1) << 63;
    uint64_t s = sign(d) << 63;
    uint64_t e = exponent(d) << 52;
    uint64_t m = mantissa(d);
    return s + e + m;
  }

  constexpr double unpack_fp(uint64_t i)
  {
    // special-case 0
    if (i == 0) return 0.0;
    if (i == uint64_t(1) << 63) return -0.0;
    constexpr uint64_t mask = uint64_t(1) << 52;
    uint64_t s = i >> 63;
    uint64_t e = ((i << 1) >> 53) - 1023;  // deal with the bias
    uint64_t m = ((i << 12) >> 12) | mask; // reset the hidden bit
    double d = std::scalbn(m, e - 52);
    return (s) ? -d : d;
  }
}
