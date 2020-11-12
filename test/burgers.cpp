#include <ttl/ttl.hpp>
#include <iostream>

namespace {
/// Model parameters
constexpr ttl::Tensor nu = ttl::scalar("nu");
constexpr ttl::Tensor  c = ttl::vector("c");

/// Dependent variables
constexpr ttl::Tensor u = ttl::vector("u");

/// Indices
constexpr ttl::Index i = 'i';
constexpr ttl::Index j = 'j';;

/// System of equations.
constexpr auto   u_rhs = nu * D(u(i),i,j) - (u(i) + c(i)) * D(u(i),j);
constexpr auto burgers = ttl::system(u = u_rhs);
constexpr auto burgers3d = ttl::scalar_system<burgers, 3>;
}

int main() {
  // fmt::print("u_rhs = {:eqn}\n", u_rhs);
  // fmt::print("graph u {{\n{:dot}}}\n", u_rhs);
  auto trees = burgers3d.make_scalar_trees();
  return 0;
}
