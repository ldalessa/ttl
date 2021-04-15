#include <ttl/ttl.hpp>

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
constexpr ttl::System burgers = { u = u_rhs };
constexpr ttl::ScalarSystem<burgers, 1> burgers1d;
}

int main()
{
  fmt::print("{}\n", burgers.scalar_trees(1)[0].to_string());

  fmt::print("u = {}\n", u_rhs.to_string());
  fmt::print("graph u {{\n{}}}\n", ttl::dot(u_rhs));

  auto&& sp = burgers.simplify(u, u_rhs);
  fmt::print("{}\n", sp.to_string());
  fmt::print("graph u {{\n{}}}\n", ttl::dot(sp));

  for (int i = 0; auto&& tree : burgers.scalar_trees(2))
  {
    fmt::print("{}: {}\n", i++, tree.to_string());
  }

  constexpr auto shapes = burgers.shapes(3);
  shapes([](auto const&... shape) {
    (fmt::print("node_depth:{} node_count:{} index_size:{} stack_depth:{}\n",
                shape.node_depth,
                shape.node_count,
                shape.index_size,
                shape.stack_depth), ...);
  });
  return 0;
}
