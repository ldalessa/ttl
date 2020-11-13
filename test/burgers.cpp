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
constexpr auto burgers2d = ttl::scalar_system<burgers, 2>;
}

int main() {
  // fmt::print("u_rhs = {:eqn}\n", u_rhs);
  for (int i = 0; auto p : burgers2d.scalars) {
    fmt::print("{}: {}\n", i++, p);
  }
  fmt::print("graph u {{\n{}}}\n", ttl::dot(std::get<0>(burgers2d.simple)));
  auto trees = burgers2d.make_scalar_trees();
  fmt::print("graph u {{\n{}}}\n", std::get<0>(trees));
  return 0;
}
