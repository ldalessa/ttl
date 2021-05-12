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

constexpr auto foo()
{
  ttl::TensorIndex i('i');
  ttl::TensorIndex j('j');
  auto a = ttl::δ(i,j);
  auto b = ttl::δ(i,j);
  auto c = a * b;
  auto d = c + 1;
  auto e = +c;
  auto f = e - 2;
  auto g = -f;
  auto h = abs(g);
  auto k = h * a;
  auto l = k(j,i);
  return l;
}

// constexpr ttl::SerializedTree tree = foo();

int main()
{
  ttl::SerializedTree tree = foo();
  ttl::print(tree, stdout);
  puts("");
  ttl::dot(tree, stdout);
  puts("");

  // test.simplify_equations()([](auto const&... eqn) {
  //   (eqn.print(stdout), ...);
  // });

  // test.simplify_equations()([](auto const&... eqn) {
  //   (eqn.dot(stdout), ...);
  // });
  return 0;
}
