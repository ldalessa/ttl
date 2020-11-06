#pragma once

#include "Index.hpp"
#include "Rational.hpp"
#include "Tensor.hpp"

namespace ttl {
enum Tag {
  SUM,
  DIFFERENCE,
  PRODUCT,
  INVERSE,
  BIND,
  PARTIAL,
  INDEX,
  DELTA,
  TENSOR,
  RATIONAL,
  DOUBLE
};

template <Tag> struct tag_t {};
template <Tag tag> constexpr inline tag_t<tag> tag_v = {};

constexpr bool is_binary(Tag tag) {
  return tag < INDEX;
}

union Node {
  Index     index;
  Tensor   tensor;
  Rational      q;
  double        d;

  constexpr Node()              noexcept : index() {}
  constexpr Node(Index index)   noexcept : index(index) {}
  constexpr Node(Tensor tensor) noexcept : tensor(tensor) {}
  constexpr Node(Rational q)    noexcept : q(q) {}
  constexpr Node(double d)      noexcept : d(d) {}
};

template <typename T> requires(std::same_as<std::remove_cv_t<T>, Node>)
struct TaggedNode {
  int i;
  Tag tag;
  T& node;
  constexpr TaggedNode(int i, Tag tag, T& node) : i(i), tag(tag), node(node) {}

  constexpr bool is(Tag t) const {
    return t == tag;
  }

  constexpr bool is_binary() const {
    return ttl::is_binary(tag);
  }

  constexpr bool is_leaf() const {
    return !is_binary();
  }

  constexpr decltype(auto) offset() const {
    return i;
  }

  constexpr decltype(auto) index() const {
    return (tag < TENSOR) ? &node.index : nullptr;
  }

  constexpr decltype(auto) tensor() const {
    return (tag == TENSOR) ? &node.tensor : nullptr;
  }

  constexpr decltype(auto) rational() const {
    return (tag == TENSOR) ? &node.q : nullptr;
  }

  constexpr decltype(auto) value() const {
    return (tag == TENSOR) ? &node.d : nullptr;
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
    const char* strings[] = {
      "+",                                      // SUM
      "-",                                      // DIFFERENCE
      "*",                                      // PRODUCT
      "/",                                      // INVERSE
      "()",                                     // BIND
      "dx",                                     // PARTIAL
      "",                                       // INDEX
      "",                                       // DELTA
      "",                                       // TENSOR
      "",                                       // RATIONAL
      ""                                        // DOUBLE
    };

    return format_to(ctx.out(), "{}", strings[tag]);
  }
};

template <typename T>
struct fmt::formatter<ttl::TaggedNode<T>> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::TaggedNode<T>& node, FormatContext& ctx) {
    switch (node.tag) {
     case ttl::SUM:
     case ttl::DIFFERENCE:
     case ttl::PRODUCT:
     case ttl::INVERSE:
     case ttl::BIND:
     case ttl::PARTIAL:    return format_to(ctx.out(), "{}", node.tag);
     case ttl::INDEX:
     case ttl::DELTA:      return format_to(ctx.out(), "{}", node.node.index);
     case ttl::TENSOR:     return format_to(ctx.out(), "{}", node.node.tensor);
     case ttl::RATIONAL:   return format_to(ctx.out(), "{}", node.node.q);
     case ttl::DOUBLE:     return format_to(ctx.out(), "{}", node.node.d);
     default:
      __builtin_unreachable();
    }
  }
};
