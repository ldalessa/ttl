#pragma once

#include "Tensor.hpp"
#include "concepts.hpp"

namespace ttl {
template <typename RHS>
requires(is_tree<RHS>)
struct Equation {
  Tensor lhs;
  RHS rhs;
  constexpr static std::true_type is_equation_tag = {};

  constexpr Equation(Tensor lhs, RHS rhs) : lhs(lhs), rhs(rhs) {
  }
};

constexpr auto
Tensor::operator=(is_tree auto&& rhs) const
{
  return Equation(*this, rhs);
}
}
