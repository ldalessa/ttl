#pragma once

#include "Index.hpp"
#include "Nodes.hpp"
#include "Rational.hpp"
#include "Tensor.hpp"
#include <functional>

namespace ttl {

union NodeData {
  struct {} _;
  Index idx;
  std::reference_wrapper<const Tensor> a;
  Rational q;
  double   d;
};

template <char N>
constexpr void copy(NodeData& lhs, const NodeData& rhs) {
  if constexpr (N == PARTIAL || N == DELTA || N == BIND) {
    lhs.i = rhs.i;
  }
  else if constexpr (N == TENSOR) {
    lhs.t = rhs.t;
  }
  else if constexpr (N == RATIONAL) {
    lhs.q = rhs.q;
  }
  else if constexpr (N == DOUBLE) {
    lhs.q = rhs.d;
  }
  else {
    lhs._ = rhs._;
  }
}

template <char... Ts>
class Tree
{
  static constexpr int M = sizeof...(Ts);
  NodeData data_[M];
 public:
  constexpr Tree() = default;

  template <Binary Node, char... As, char... Bs>
  constexpr Tree(Node, Tree<As...> a, Tree<Bs...> B);

  template <Unary Node, char... As>
  constexpr Tree(Node, Index i, Tree<As...> a) {
    data_[0].i = i;
    ((assign
  }

  constexpr Tree(Node<DELTA>, Index i) {
    data_[0].i = i;
  }

  constexpr Tree(Node<TENSOR>, const Tensor& t) {
    data_[0].t = std::cref(t);
  }

  constexpr Tree(Node<RATIONAL>, Rational q) {
    data_[0].q = q;
  }

  constexpr Tree(Node<DOUBLE>, double d) {
    data_[0].d = d;
  }
};

template <Binary Node, char... As, char... Bs>
Tree(Node, Tree<As...>, Tree<Bs...>) -> Tree<node_type_v<Node>, As..., Bs...>;

template <Unary Node, char... As>
Tree(Node, Index, Tree<As...>) -> Tree<node_type_v<Node>, As...>;

template <Leaf Node, typename T>
Tree(Node, T) -> Tree<node_type_v<Node>>;
}
