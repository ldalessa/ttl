#include <ttl/ttl.hpp>
#include <iostream>

template <ttl::Node Density, ttl::Node Energy, ttl::Node T>
static constexpr auto ideal_gas(Density&& rho, Energy&& e, T&& gamma) {
  return (gamma - 1) * rho * e;
}

template <ttl::Node Pressure, ttl::Node Velocity, ttl::Node T, ttl::Node U>
static constexpr auto newtonian_fluid(Pressure&& p, Velocity&& v, T&& mu, U&& muVolume) {
  ttl::index i = ttl::idx<'a'>;
  ttl::index j = ttl::idx<'b'>;
  ttl::index k = ttl::idx<'c'>;

  auto   d = symmetrize(D(v(i),j));
  auto iso = p + muVolume * D(v(k),k);
  auto dev = 2 * mu * d - 2.0 / 3.0 * mu * D(v(k),k) * delta(i,j);
  return iso * delta(i,j) + dev;
}

template <ttl::Node Energy, ttl::Node T>
static constexpr auto calorically_perfect(Energy&& e, T&& specific_heat) {
  return e / specific_heat;
}

template <ttl::Node Temperature, ttl::Node T>
static constexpr auto fouriers_law(Temperature&& theta, T&& conductivity) {
  ttl::index i = ttl::idx<'c'>;
  return - D(theta,i) * conductivity;
}

/// 1. Model parameters.
static constexpr ttl::tensor    gamma = ttl::scalar("gamma");
static constexpr ttl::tensor       mu = ttl::scalar("mu");
static constexpr ttl::tensor muVolume = ttl::scalar("muVolume");
static constexpr ttl::tensor       cv = ttl::scalar("cv");
static constexpr ttl::tensor    kappa = ttl::scalar("kappa");
static constexpr ttl::tensor        g = ttl::vector("g");

/// 2. Dependent variables.
static constexpr ttl::tensor rho = ttl::scalar("rho");
static constexpr ttl::tensor   e = ttl::scalar("e");
static constexpr ttl::tensor   v = ttl::vector("v");

/// 3. Tensor indices.
static constexpr ttl::index i = ttl::idx<'i'>;
static constexpr ttl::index j = ttl::idx<'j'>;

/// 4. Constitutive model terms.
static constexpr auto     d = symmetrize(D(v(i),j));
static constexpr auto     p = ideal_gas(rho, e, gamma);
static constexpr auto sigma = newtonian_fluid(p, v, mu, muVolume);
static constexpr auto theta = calorically_perfect(e, cv);
static constexpr auto     q = fouriers_law(theta, kappa);

/// 5. Update equations.
static constexpr auto rho_rhs = - D(rho,i) * v(i) - rho * D(v(i),i);
static constexpr auto   v_rhs = - D(v(i),j) * v(j) + D(sigma(i,j),j) / rho + g(i);
static constexpr auto   e_rhs = - v(i) * D(e,i) + sigma(i,j) * d(i,j) / rho - D(q(i),i) / rho;

int main() {
  std::cout << ttl::dot("d") << d << "\n";
  std::cout << ttl::dot("p") << p << "\n";
  std::cout << ttl::dot("sigma") << sigma << "\n";
  std::cout << ttl::dot("theta") << theta << "\n";
  std::cout << ttl::dot("q") << q << "\n";

  std::cout << ttl::dot("rho") << rho_rhs << "\n";
  std::cout << ttl::dot("v") << v_rhs << "\n";
  std::cout << ttl::dot("e") << e_rhs << "\n";
  return 0;
}
