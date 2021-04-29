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
  constexpr ttl::System burgers = { u <<= u_rhs };
  // constexpr ttl::ExecutableSystem<double, 1, burgers> burgers1d;
}

int main()
{
  burgers.equations([](ttl::is_equation auto const&... eqn) {
    (eqn([](ttl::Tensor const *lhs, const auto& rhs) {
      fmt::print("{} = {}\n", *lhs, to_string(rhs));
    }), ...);
  });

  constexpr auto b = burgers.simplify_equations(1);

  burgers.simplify_equations()([](auto const&... eqn) {
    (eqn.print(stdout), ...);
  });

  burgers.equations([](ttl::is_equation auto const&... eqn) {
    (eqn.dot(stdout), ...);
  });

  return 0;
}
