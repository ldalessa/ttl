#pragma once

#include <tuple>

namespace ttl {
constexpr auto make_system_of_equations(auto&&... eqns) {
  return std::tuple(std::forward<decltype(eqns)>(eqns)...);
}
}
