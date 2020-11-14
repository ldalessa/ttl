#pragma once

#include "ttl/Index.hpp"
#include <fmt/core.h>

namespace ttl::tensor {
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
   default:
    assert(false);
  }
}

constexpr bool tag_is_binary(Tag tag) {
  return tag < INDEX;
}
}

template <>
struct fmt::formatter<ttl::tensor::Tag>
{
  constexpr static const char* tag_strings[] = {
    "+",                                        // SUM
    "-",                                        // DIFFERENCE
    "*",                                        // PRODUCT
    "/",                                        // RATIO
    "dx",                                       // PARTIAL
    "",                                         // INDEX
    "",                                         // TENSOR
    "",                                         // RATIONAL
    ""                                          // DOUBLE
  };

  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(ttl::tensor::Tag tag, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", tag_strings[tag]);
  }
};
