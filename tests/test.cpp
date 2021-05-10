#include <ttl/ttl.hpp>
#include <cassert>

namespace {
  // constexpr ttl::Tensor A = ttl::matrix("A");
  // constexpr ttl::Tensor B = ttl::matrix("B");
  // constexpr ttl::Tensor C = ttl::matrix("C");
  // constexpr ttl::Tensor a = ttl::vector("a");
  // constexpr ttl::Tensor b = ttl::scalar("b");

  // constexpr ttl::Index i = 'i';
  // constexpr ttl::Index j = 'j';

  // constexpr auto source = ttl::scalar(
  //   B(0,0) + B(1,1) + A(i,j)*B(i,j),
  //   0);

  // constexpr auto c = A(i,j) * source(j);

  // constexpr ttl::System test = {
  //   b <<= b_rhs
  // };

  // constexpr ttl::ExecutableSystem<double, 3, test> test3d;
}


constexpr bool foo()
{
  ttl::TensorIndex i('i');
  ttl::TensorIndex j('j');
  auto a = ttl::δ(i,j);
  auto b = ttl::δ(i,j);
  auto c = abs(-(+(a * b + 1) - 2)) * a;
  auto d = c(j,i);
  // assert(ttl::outer_index(a) == (i + j));
  // ttl::LinkedTree<1> b(2);
  // auto c = a * a;
  // auto c = -(a + a*b - a);
  // assert(ttl::outer_index(c) == (i + j));
  // auto d = c + c;
  // assert(ttl::outer_index(d) == (i + j));
  // auto e = c * d / a;
  // assert(ttl::outer_index(e) == (i + j));
  // auto f = D(e, i, j);
  // assert(ttl::outer_index(f) == ttl::TensorIndex{});
  // auto g = abs(f);
  // auto h = a ** g;
  // auto k = fmin(h * a, 1) * ttl::δ(i,j);
  // auto l = k(j,i);
  return true;
}

static_assert(foo());

int main()
{
  foo();
  // test.simplify_equations()([](auto const&... eqn) {
  //   (eqn.print(stdout), ...);
  // });

  // test.simplify_equations()([](auto const&... eqn) {
  //   (eqn.dot(stdout), ...);
  // });
  return 0;
}
