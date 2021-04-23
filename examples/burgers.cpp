#include <ttl/ttl.hpp>

namespace {
  /// Model parameters
  constexpr ttl::Tensor ν = ttl::scalar("ν");
  constexpr ttl::Tensor c = ttl::vector("c");

  /// Dependent variables
  constexpr ttl::Tensor u = ttl::vector("u");

  /// Indices
  constexpr ttl::Index i = 'i';
  constexpr ttl::Index j = 'j';;

  /// System of equations.
  constexpr auto   u_rhs = ν * D(u(i),i,j) - (u(i) + c(i)) * D(u(i),j);
  // constexpr ttl::System burgers = { u <<= u_rhs };
  // constexpr ttl::ExecutableSystem<double, 1, burgers> burgers1d;
}

int main()
{
  return 0;
}
