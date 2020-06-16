#pragma once

#include "expression.hpp"
#include "mp/cvector.hpp"

namespace ttl {
template <Cardinality<2>... Equations>
constexpr int make_system_of_equations(Equations&&... eqns) {

  return 0;
}
}
