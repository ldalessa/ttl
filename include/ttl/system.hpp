#pragma once

#include "Equation.hpp"
#include "tensor/System.hpp"

namespace ttl
{
template <typename... Equations>
requires(is_equation<Equations> && ...)
constexpr auto system(Equations&&... eqns) {
  return tensor::System(std::forward<Equations>(eqns)...);
}
}
