#include <ttl/ttl.hpp>
#include <iostream>

/// Model parameters
static constexpr ttl::tensor nu = ttl::scalar("nu");
static constexpr ttl::tensor  c = ttl::vector("c");

/// Dependent variables
static constexpr ttl::tensor u = ttl::vector("u");

/// Indices
static constexpr ttl::index i = ttl::idx<'i'>;
static constexpr ttl::index j = ttl::idx<'j'>;;

/// 5. Our update equations and their corresponding epsilon definitions.
static constexpr auto u_rhs = nu * D(u(i),i,j) - (u(i) + c(i)) * D(u(i),j);

int main() {
  std::cout << ttl::dot("u") << u_rhs << "\n";
}
