#pragma once

#include "Tensor.hpp"
#include "Tree.hpp"

namespace ttl {
template <typename RHS>
requires(is_tree<RHS>)
struct Equation {
  Tensor lhs;
  RHS rhs;

  constexpr friend void is_equation_trait(Equation) {}

  constexpr Equation(Tensor lhs, RHS rhs) : lhs(lhs), rhs(rhs) {
  }
};

template <typename T>
concept is_equation = requires(T t) {
  { is_equation_trait(t) };
};

template <int M>
constexpr auto
Tensor::operator=(const Tree<M>& rhs) const {
  return Equation(*this, rhs);
}

template <int M>
constexpr auto
Tensor::operator=(Tree<M>&& rhs) const {
  return Equation(*this, std::move(rhs));
}
}
