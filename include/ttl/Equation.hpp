#pragma once

#include "Tensor.hpp"
#include "concepts.hpp"

namespace ttl {
template <is_tree Tree>
struct Equation {
  constexpr static std::true_type is_equation_tag = {};

  const Tensor* lhs;
  Tree rhs;

  constexpr Equation(const Tensor* lhs, Tree rhs) : lhs(lhs), rhs(rhs) {}
};

template <is_tree Tree>
constexpr auto Tensor::operator=(Tree&& rhs) const {
  assert(order_ == rhs.outer().size());
  return Equation(this, std::forward<Tree>(rhs));
}
}
