#pragma once

#include "Index.hpp"
#include "Rational.hpp"
#include "Tensor.hpp"
#include <fmt/core.h>

namespace ttl {
enum Tag {
  SUM,
  DIFFERENCE,
  PRODUCT,
  RATIO,
  BIND,
  PARTIAL,
  INDEX,
  DELTA,
  TENSOR,
  RATIONAL,
  DOUBLE
};

constexpr bool binary(Tag tag) {
  return tag < INDEX;
}

constexpr bool leaf(Tag tag) {
  return INDEX <= tag;
}

constexpr bool index(Tag tag) {
  return tag < TENSOR;
}

constexpr Index outer(Tag tag, const Index& a, const Index& b) {
  assert(binary(tag));
  switch (tag) {
   case SUM:
   case DIFFERENCE:
    assert(permutation(a, b));
    return a;
   case PRODUCT:
   case RATIO:
    return a ^ b;
   case BIND:
   case PARTIAL:
    return exclusive(a + b);
   default:
    __builtin_unreachable();
  }
}

constexpr static const char* tag_strings[] = {
  "+",                                        // SUM
  "-",                                        // DIFFERENCE
  "*",                                        // PRODUCT
  "/",                                        // RATIO
  "()",                                       // BIND
  "dx",                                       // PARTIAL
  "",                                         // INDEX
  "",                                         // DELTA
  "",                                         // TENSOR
  "",                                         // RATIONAL
  ""                                          // DOUBLE
};

constexpr const char* to_string(Tag tag) {
  return tag_strings[tag];
}

union Data {
  const Tensor* tensor = nullptr;        // need to pick something for constexpr
  Index index;
  Rational q;
  double d;

  constexpr Data() {}

  constexpr Data(const Tensor* t)
      : tensor(t)
  {
  }

  constexpr Data(const Index& i)
      : index(i)
  {
  }

  constexpr Data(const Rational& q)
      : q(q)
  {
  }

  constexpr Data(double d)
      : d(d)
  {
  }
};

struct Node {
  Tag     tag;
  Data   data = {};
  // Index   idx = {};

  constexpr Node() {}
  constexpr Node(const Tensor* tensor)
      : tag(TENSOR)
      , data(tensor)
  {
  }

  constexpr Node(Rational q)
      : tag(RATIONAL)
      , data(q)
  {
  }

  constexpr Node(double d)
      : tag(DOUBLE)
      , data(d)
  {
  }

  constexpr Node(Index i, Tag tag)
      : tag(tag)
      , data(i)
  {
  }

  constexpr bool binary() const {
    return ttl::binary(tag);
  }

  constexpr const Tensor* tensor() const {
    return (tag == TENSOR) ? data.tensor : nullptr;
  }

  constexpr const Index* index() const {
    return (ttl::index(tag)) ? &data.index : nullptr;
  }

  constexpr Index* index() {
    return (ttl::index(tag)) ? &data.index : nullptr;
  }
};
}

template <>
struct fmt::formatter<ttl::Tag> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(ttl::Tag tag, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", ttl::tag_strings[tag]);
  }
};

template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<ttl::Node, T>, char>> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const T& node, FormatContext& ctx) {
    switch (node.tag) {
     case ttl::SUM:
     case ttl::DIFFERENCE:
     case ttl::PRODUCT:
     case ttl::RATIO:
     case ttl::BIND:
     case ttl::PARTIAL:    return format_to(ctx.out(), "{}", node.tag);
     case ttl::INDEX:
     case ttl::DELTA:      return format_to(ctx.out(), "{}", node.data.index);
     case ttl::TENSOR:     return format_to(ctx.out(), "{}", *node.data.tensor);
     case ttl::RATIONAL:   return format_to(ctx.out(), "{}", node.data.q);
     case ttl::DOUBLE:     return format_to(ctx.out(), "{}", node.data.d);
     default:
      __builtin_unreachable();
    }
  }
};
