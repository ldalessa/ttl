#include <ttl/ttl.hpp>

namespace {
  constexpr ttl::Tensor A = ttl::matrix("A");
  constexpr ttl::Tensor B = ttl::matrix("B");
  constexpr ttl::Tensor C = ttl::matrix("C");
  constexpr ttl::Tensor a = ttl::vector("a");

  constexpr ttl::Index i = 'i';
  constexpr ttl::Index j = 'j';

  // constexpr ttl::System test = {
  //   A <<= B(i,j),
  //   B <<= C(i,j),
  //   a <<= D(ttl::exp(A(i,j) + B(i,j)),j)
  // };

  constexpr ttl::System test = {
    B <<= A(i,j),
    a <<= D(ttl::sqrt(B(i,j)), j)
  };

  // constexpr ttl::ExecutableSystem<double, 3, test> test3d;
}

int main()
{
  test.simplify_equations()([](auto const&... eqn) {
    (eqn.print(stdout), ...);
  });

  test.simplify_equations()([](auto const&... eqn) {
    (eqn.dot(stdout), ...);
  });

  return 0;
}
