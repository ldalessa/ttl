#pragma once

#include "ttl/concepts.hpp"
#include <bit>

namespace ttl
{
  using tag_id_t = unsigned;

  enum Tag : tag_id_t {
    SUM        = 1lu << 0,
    DIFFERENCE = 1lu << 1,
    PRODUCT    = 1lu << 2,
    RATIO      = 1lu << 3,
    POW        = 1lu << 4,
    BIND       = 1lu << 5,
    PARTIAL    = 1lu << 6,
    SQRT       = 1lu << 7,
    EXP        = 1lu << 8,
    NEGATE     = 1lu << 9,
    RATIONAL   = 1lu << 10,
    DOUBLE     = 1lu << 11,
    TENSOR     = 1lu << 12,
    SCALAR     = 1lu << 13,
    DELTA      = 1lu << 14,
    EPSILON    = 1lu << 15,
    MAX        = 1lu << 16
  };

  constexpr inline Tag NO_TAG{0};

  constexpr auto operator|(Tag a, Tag b) -> Tag
  {
    return Tag(tag_id_t(a) | tag_id_t(b));
  }

  constexpr auto operator&(Tag a, Tag b) -> Tag
  {
    return Tag(tag_id_t(a) & tag_id_t(b));
  }

  constexpr inline Tag ALL = Tag(MAX - 1);
  constexpr inline Tag BINARY = SUM | DIFFERENCE | PRODUCT | RATIO | POW;
  constexpr inline Tag UNARY = BIND | PARTIAL | SQRT | EXP | NEGATE;
  constexpr inline Tag LEAF = RATIONAL | DOUBLE | TENSOR | SCALAR | DELTA | EPSILON;
  constexpr inline Tag LITERAL = RATIONAL | DOUBLE;
  constexpr inline Tag ADDITION = SUM | DIFFERENCE;
  constexpr inline Tag MULTIPLICATION = PRODUCT | RATIO;

  constexpr auto ispow2(Tag tag)
  {
    return std::has_single_bit(tag_id_t(tag));
  }

  constexpr auto popcount(Tag tag)
  {
    return std::popcount(tag_id_t(tag));
  }

  constexpr auto ctz(Tag tag)
  {
    return std::countr_zero(tag_id_t(tag));
  }

  constexpr auto tag_is_simple(Tag tag)
  {
    return ispow2(tag);
  }

  constexpr auto tag_is_composite(Tag tag)
  {
    return popcount(tag) > 1;
  }

  constexpr auto tag_to_index(Tag tag) -> int
  {
    assert(tag_is_simple(tag));
    return ctz(tag);
  }

  constexpr bool tag_is(Tag tag, Tag set)
  {
    return tag_is_simple(tag & set);
  }

  constexpr bool tag_is_binary(Tag tag)
  {
    return tag_is(tag, BINARY);
  }

  constexpr bool tag_is_unary(Tag tag)
  {
    return tag_is(tag, UNARY);
  }

  constexpr bool tag_is_leaf(Tag tag)
  {
    return tag_is(tag, LEAF);
  }

  constexpr bool tag_is_literal(Tag tag)
  {
    return tag_is(tag, LITERAL);
  }

  constexpr bool tag_is_multiplication(Tag tag)
  {
    return tag_is(tag, MULTIPLICATION);
  }

  constexpr bool tag_is_addition(Tag tag)
  {
    return tag_is(tag, ADDITION);
  }

  template <class T>
  constexpr auto tag_outer(Tag tag, T const& a, T const& b = {}) -> T
  {
    if (tag_is_addition(tag)) {
      return a;
    }

    if (tag_is_multiplication(tag)) {
      return a ^ b;
    }

    if (tag == PARTIAL) {
      return exclusive(a + b);
    }

    if (tag == BIND) {
      return b;
    }

    return exclusive(a);
  }

  constexpr auto tag_eval(Tag tag, auto const& a, auto const& b)
  {
    assert(tag_is_binary(tag));

    using std::pow;
    switch (tag) {
     case SUM:        return a + b;
     case DIFFERENCE: return a - b;
     case PRODUCT:    return a * b;
     case RATIO:      return a / b;
     case POW:        return pow(a, b);
     default:
      assert(false);
    }
    __builtin_unreachable();
  }

  constexpr auto tag_to_string(Tag tag) -> const char*
  {
    constexpr const char* strings[] = {
      "+", // SUM
      "-", // DIFFERENCE
      "*", // PRODUCT
      "/", // RATIO
      "^", // POW
      "bind",  // BIND
      "∂", // PARTIAL
      "√", // SQRT
      "ℇ", // EXP
      "-", // NEGATE
      "",  // RATIONAL
      "",  // DOUBLE
      "",  // TENSOR
      "",  // SCALAR
      "δ", // DELTA
      "ε", // EPSILON
    };
    static_assert(std::size(strings) == tag_to_index(MAX));
    return strings[tag_to_index(tag)];
  }
}

#include <fmt/format.h>

template <>
struct fmt::formatter<ttl::Tag>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  constexpr auto format(ttl::Tag tag, auto& ctx)
  {
    return fmt::format_to(ctx.out(), "{}", tag_to_string(tag));
  }
};
