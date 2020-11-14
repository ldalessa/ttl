#pragma once

#include "Rational.hpp"
#include "Tensor.hpp"
#include "tensor/RPNTree.hpp"
#include "concepts.hpp"

namespace ttl::tensor {
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
constexpr RPNTree<M>&& bind(RPNTree<M>&& a) {
  return std::move(a);
}

template <int M>
constexpr const RPNTree<M>& bind(const RPNTree<M>& a) {
  return a;
}

constexpr RPNTree<1> bind(is_expr auto const& a) {
  return RPNTree(a);
}

constexpr auto operator+(is_expr auto const& a) {
  return bind(a);
}

constexpr auto operator+(is_expr auto const& a, is_expr auto const& b) {
  return RPNTree(SUM, bind(a), bind(b));
}

constexpr auto operator*(is_expr auto const& a, is_expr auto const& b) {
  return RPNTree(PRODUCT, bind(a), bind(b));
}

constexpr auto operator-(is_expr auto const& a, is_expr auto const& b) {
  return RPNTree(DIFFERENCE, bind(a), bind(b));
}

constexpr auto operator-(is_expr auto const& a) {
  return RPNTree(-1) * bind(a);
}

constexpr auto operator/(is_expr auto const& a, is_expr auto const& b) {
  return RPNTree(RATIO, bind(a), bind(b));
}

constexpr auto D(is_expr auto const& a, std::same_as<Index> auto... is) {
  return RPNTree(PARTIAL, bind(a), RPNTree((is + ...)));
}

constexpr RPNTree<1> delta(const Index& a, const Index& b) {
  assert(a.size() == 1);
  assert(b.size() == 1);
  assert(a != b);
  return RPNTree(a + b);
}

constexpr auto symmetrize(is_expr auto const& a) {
  RPNTree t = bind(a);
  return RPNTree(Rational(1,2)) * (t + t(reverse(t.outer())));
}
}

namespace ttl {
constexpr auto Tensor::operator()(std::same_as<Index> auto... is) const {
  return ttl::tensor::RPNTree(*this, (is + ... + Index()));
}
}
