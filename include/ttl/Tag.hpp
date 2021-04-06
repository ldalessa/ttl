#pragma once

#include "Index.hpp"
#include <fmt/core.h>

namespace ttl {
enum Tag {
  SUM,
  DIFFERENCE,
  PRODUCT,
  RATIO,
  PARTIAL,
  INDEX,
  TENSOR,
  RATIONAL,
  DOUBLE
};

constexpr Index tag_outer(Tag tag, const Index& a, const Index& b) {
  switch (tag) {
   case SUM:
   case DIFFERENCE:
    assert(permutation(a, b));
    return a;

   case PRODUCT:
   case RATIO:
    return a ^ b;

   case PARTIAL:
    return exclusive(a + b);

   default: assert(false);
  }
  __builtin_unreachable();
}

template <typename T>
constexpr auto tag_apply(Tag tag, T&& a, T&& b) {
  switch (tag) {
   case SUM:        return std::forward<T>(a) + std::forward<T>(b);
   case DIFFERENCE: return std::forward<T>(a) - std::forward<T>(b);
   case PRODUCT:    return std::forward<T>(a) * std::forward<T>(b);
   case RATIO:      return std::forward<T>(a) / std::forward<T>(b);
   default: assert(false);
  }
  __builtin_unreachable();
}

constexpr bool tag_is_binary(Tag tag) {
  return tag < INDEX;
}
}

template <>
struct fmt::formatter<ttl::Tag>
{
  constexpr static const char* tag_strings[] = {
    "+",                                        // SUM
    "-",                                        // DIFFERENCE
    "*",                                        // PRODUCT
    "/",                                        // RATIO
    "∂",                                        // PARTIAL
    "",                                         // INDEX
    "",                                         // TENSOR
    "",                                         // RATIONAL
    ""                                          // DOUBLE
  };

  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(ttl::Tag tag, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", tag_strings[tag]);
  }
};
