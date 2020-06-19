#include <ttl2/tree.hpp>
#include <ttl2/dot.hpp>
#include <iostream>

namespace {
/// Model parameters
constexpr ttl::tensor nu = { 0, "nu" };
constexpr ttl::tensor  c = { 1, "c" };

/// Dependent variables
constexpr ttl::tensor  u = {1, "u" };

// /// Indices
constexpr ttl::index i = 'i';
constexpr ttl::index j = 'j';

// /// Update equations
constexpr auto u_rhs = nu * D(u(i),i,j) - (u(i) + c(i)) * D(u(i),j);

// /// Boilerplate
// constexpr auto system = ttl::make_system_of_equations(std::tie(u, u_rhs));
}

int main() {
  std::cout << ttl::dot("u") << u_rhs << "\n";
  return 0;
}
