#include <ttl/ttl.hpp>

namespace {
  constexpr ttl::Tensor A = ttl::matrix("A");
  constexpr ttl::Tensor B = ttl::matrix("B");
  constexpr ttl::Tensor C = ttl::matrix("C");
  constexpr ttl::Tensor a = ttl::vector("a");
  constexpr ttl::Tensor b = ttl::scalar("b");

  constexpr ttl::Index i = 'i';
  constexpr ttl::Index j = 'j';

  constexpr auto source = ttl::scalar(
    B(0,0) + B(1,1) + A(i,j)*B(i,j),
    0);

  constexpr auto c = A(i,j) * source(j);

  // constexpr ttl::System test = {
  //   b <<= b_rhs
  // };

  // constexpr ttl::ExecutableSystem<double, 3, test> test3d;
}

int main()
{
  // test.simplify_equations()([](auto const&... eqn) {
  //   (eqn.print(stdout), ...);
  // });

  // test.simplify_equations()([](auto const&... eqn) {
  //   (eqn.dot(stdout), ...);
  // });

  return 0;
}
