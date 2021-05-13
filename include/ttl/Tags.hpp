#pragma once

#include <bit>

namespace ttl
{
  enum TreeTag : unsigned {
    SUM          = 1lu << 0,
    DIFFERENCE   = 1lu << 1,
    PRODUCT      = 1lu << 2,
    RATIO        = 1lu << 3,
    BIND         = 1lu << 4,
    NEGATE       = 1lu << 5,
    EXPONENT     = 1lu << 6,
    PARTIAL      = 1lu << 7,
    CMATH        = 1lu << 8,
    LITERAL      = 1lu << 9,
    TENSOR       = 1lu << 10,
    SCALAR       = 1lu << 11,
    DELTA        = 1lu << 12,
    EPSILON      = 1lu << 13
  };

  constexpr TreeTag TREE_TAG_MAX = TreeTag(unsigned(EPSILON) << 1);

  constexpr auto ctz(TreeTag tag)
  {
    return std::countr_zero(unsigned(tag));
  }

  constexpr auto to_index(TreeTag tag)
  {
    return ctz(tag);
  }

  enum CMathTag : unsigned {
    ABS,
    FMIN,
    FMAX,
    EXP,
    LOG,
    POW,
    SQRT,
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    ATAN2,
    SINH,
    COSH,
    TANH,
    ASINH,
    ACOSH,
    ATANH,
    CEIL,
    FLOOR
  };

  constexpr CMathTag CMATH_TAG_MAX = CMathTag(unsigned(FLOOR) + 1);

  constexpr auto to_index(CMathTag tag)
  {
    return tag;
  }
}

#include <fmt/format.h>

template <>
struct fmt::formatter<ttl::TreeTag>
{
  constexpr static const char* strings[] = {
    "+",  // SUM
    "-",  // DIFFERENCE
    "*",  // PRODUCT
    "/",  // RATIO
    "",   // BIND
    "-",  // NEGATE
    "**", // EXPONENT
    "∂",  // PARTIAL
    "",   // CMATH
    "",   // LITERAL
    "",   // TENSOR
    "",   // SCALAR
    "δ",  // DELTA
    "ε",  // EPSILON
  };

  static_assert(std::size(strings) == ttl::to_index(ttl::TREE_TAG_MAX));

  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  constexpr auto format(ttl::TreeTag tag, auto& ctx)
  {
    return fmt::format_to(ctx.out(), "{}", strings[to_index(tag)]);
  }
};


template <>
struct fmt::formatter<ttl::CMathTag>
{
  constexpr static const char* strings[] = {
    "abs",   // ABS,
    "fmin",  // FMIN,
    "fmax",  // FMAX,
    "exp",   // EXP,
    "log",   // LOG,
    "pow",   // POW,
    "sqrt",  // SQRT,
    "sin",   // SIN,
    "cos",   // COS,
    "tan",   // TAN,
    "asin",  // ASIN,
    "acos",  // ACOS,
    "atan",  // ATAN,
    "atan2", // ATAN2,
    "sinh",  // SINH,
    "cosh",  // COSH,
    "tanh",  // TANH,
    "asinh", // ASINH,
    "acosh", // ACOSH,
    "atanh", // ATANH,
    "ceil",  // CEIL,
    "floor", // FLOOR
  };

  static_assert(std::size(strings) == ttl::to_index(ttl::CMATH_TAG_MAX));

  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  constexpr auto format(ttl::CMathTag tag, auto& ctx)
  {
    return fmt::format_to(ctx.out(), "{}", strings[to_index(tag)]);
  }
};
