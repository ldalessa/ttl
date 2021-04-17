#include <ttl/ttl.hpp>

namespace {
  constexpr ttl::Tensor A = ttl::matrix("A");
  constexpr ttl::Tensor B = ttl::matrix("B");
  constexpr ttl::Tensor C = ttl::matrix("C");

  constexpr ttl::Index i = 'i';
  constexpr ttl::Index j = 'j';

  constexpr ttl::System test = {
    C = A(i,j) + B(i,j)
  };

  constexpr ttl::ScalarSystem<test, 3> test3d;
}

int main()
{
  puts("scalars");
  for (auto&& s : test3d.scalars) {
    fmt::print("\t{}\n", s.to_string());
  }

  puts("constants");
  for (auto&& s : test3d.constants) {
    fmt::print("\t{}\n", s.to_string());
  }

  puts("parse trees");
  test.equations([](ttl::is_equation auto const&... eqns) {
    (fmt::print("\t{} = {}\n", eqns.lhs, eqns.rhs.to_string()), ...);
  });

  puts("tensor trees");
  // auto trees = test.simplify_trees();
  // trees([](auto const&... tree) {
  //   (fmt::print("\t{}\n", tree.to_string()), ...);
// });

  puts("scalar trees");
  for (auto&& tree : test.scalar_trees(3)) {
    fmt::print("\t{}\n", tree.to_string());
  }

  constexpr auto shapes = test.shapes(3);
  shapes([](auto const&... shape) {
    (fmt::print("{}\n", shape), ...);
  });
  return 0;
}
