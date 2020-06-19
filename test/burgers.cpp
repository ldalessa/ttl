#include <ttl/ttl.hpp>
#include <iostream>

namespace {
/// Model parameters
constexpr ttl::tensor nu = ttl::scalar("nu");
constexpr ttl::tensor  c = ttl::vector("c");

/// Dependent variables
constexpr ttl::tensor u = ttl::vector("u");

/// Indices
constexpr ttl::index i = ttl::idx<'i'>;
constexpr ttl::index j = ttl::idx<'j'>;;

/// Update equations
constexpr auto u_rhs = nu * D(u(i),i,j) - (u(i) + c(i)) * D(u(i),j);

/// Boilerplate
// constexpr auto system = ttl::make_system_of_equations(std::tie(u, u_rhs));
}

int main() {
  std::cout << ttl::dot("u") << u_rhs << "\n";
  return 0;
}
