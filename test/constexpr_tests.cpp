#include <ttl/ttl.hpp>
#include <iostream>

constexpr ttl::tensor rho(0, "rho");
constexpr ttl::tensor v(1, "v");
constexpr ttl::index i('i');
constexpr ttl::index j('j');
constexpr ttl::index ij = i + j;
constexpr ttl::index jj = ij ^ i;
constexpr ttl::bind bs(rho);
constexpr ttl::bind bv(v, jj);
constexpr ttl::inverse iv(bv);
constexpr auto dd = delta(i, j);

constexpr auto a = bv + bv;
constexpr auto b = (rho + rho) * (bv + bv);
constexpr auto c = bv * bv;
constexpr auto d = bs / bv;
constexpr auto e = D(rho, i, j);
constexpr auto f = D(dd, i, j);
constexpr auto g = D(c, i, j);
constexpr auto h = symmetrize(bv);

int main() {
  constexpr int i = order(v);
  static_assert(i == 1);
  std::cout << ttl::dot("g") << g << "\n";
  return i;
}
