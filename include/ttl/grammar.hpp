#pragma once

#include "ParseTree.hpp"
#include "Rational.hpp"
#include "Tensor.hpp"
#include "concepts.hpp"

namespace ttl {
  template <typename T>
  concept is_expr =
  is_tree<T> ||
  std::same_as<T, Tensor> ||
  std::same_as<T, Rational> ||
  std::signed_integral<T> ||
  std::floating_point<T>;

  /// Bind is used to reduce the overhead of processing the grammar. In
  /// particular, we want to support using tensors, rationals, and integral types
  /// directly in the grammar along with trees. This requires that we construct
  /// trees inside the grammar rules, but we don't want to have to make copies
  /// when we already have a tree.
  template <int M>
  constexpr ParseTree<M>&& bind(ParseTree<M>&& a) {
    return std::move(a);
  }

  template <int M>
  constexpr const ParseTree<M>& bind(const ParseTree<M>& a) {
    return a;
  }

  constexpr ParseTree<1> bind(is_expr auto const& a) {
    return ParseTree(a);
  }

  constexpr auto operator+(is_expr auto const& a) {
    return bind(a);
  }

  constexpr auto operator+(is_expr auto const& a, is_expr auto const& b) {
    return ParseTree(SUM, bind(a), bind(b));
  }

  constexpr auto operator*(is_expr auto const& a, is_expr auto const& b) {
    return ParseTree(PRODUCT, bind(a), bind(b));
  }

  constexpr auto operator-(is_expr auto const& a, is_expr auto const& b) {
    return ParseTree(DIFFERENCE, bind(a), bind(b));
  }

  constexpr auto operator-(is_expr auto const& a) {
    return ParseTree(-1) * bind(a);
  }

  constexpr auto operator/(is_expr auto const& a, is_expr auto const& b) {
    return ParseTree(RATIO, bind(a), bind(b));
  }

  constexpr auto D(is_expr auto const& a, std::same_as<Index> auto... is) {
    return ParseTree(PARTIAL, bind(a), ParseTree((is + ...)));
  }

  constexpr ParseTree<1> delta(const Index& a, const Index& b) {
    assert(a.size() == 1);
    assert(b.size() == 1);
    assert(a != b);
    return ParseTree(a + b);
  }

  constexpr auto symmetrize(is_expr auto const& a) {
    ParseTree t = bind(a);
    return ParseTree(Rational(1,2)) * (t + t(reverse(t.outer())));
  }
}
