#include <ttl/ttl.hpp>
#include <iostream>

namespace {
template <ttl::Node Density, ttl::Node Energy, ttl::Node T>
constexpr auto ideal_gas(Density&& rho, Energy&& e, T&& gamma) {
  return (gamma - 1) * rho * e;
}

template <ttl::Node Pressure, ttl::Node Velocity, ttl::Node T, ttl::Node U>
constexpr auto newtonian_fluid(Pressure&& p, Velocity&& v, T&& mu, U&& muVolume) {
  ttl::index i('a');
  ttl::index j('b');
  ttl::index k('c');

  auto   d = symmetrize(D(v(i),j));
  auto iso = p + muVolume * D(v(k),k);
  auto dev = 2 * mu * d - 2.0 / 3.0 * mu * D(v(k),k) * delta(i,j);
  return iso * delta(i,j) + dev;
}

template <ttl::Node Energy, ttl::Node T>
constexpr auto calorically_perfect(Energy&& e, T&& specific_heat) {
  return e / specific_heat;
}

template <ttl::Node Temperature, ttl::Node T>
constexpr auto fouriers_law(Temperature&& theta, T&& conductivity) {
  ttl::index i('c');
  return - D(theta,i) * conductivity;
}

/// 1. Model parameters.
constexpr ttl::tensor    gamma(0, "gamma");
constexpr ttl::tensor       mu(0, "mu");
constexpr ttl::tensor muVolume(0, "muVolume");
constexpr ttl::tensor       cv(0, "cv");
constexpr ttl::tensor    kappa(0, "kappa");
constexpr ttl::tensor        g(1, "g");

/// 2. Dependent variables.
constexpr ttl::tensor rho(0, "rho");
constexpr ttl::tensor   e(0, "e");
constexpr ttl::tensor   v(1, "v");

/// 3. Indices used in our modeling.
constexpr ttl::index i('i');
constexpr ttl::index j('j');

/// 4. Constitutive model terms.
constexpr auto     d = symmetrize(D(v(i),j));
constexpr auto     p = ideal_gas(rho, e, gamma);
constexpr auto sigma = newtonian_fluid(p, v, mu, muVolume);
constexpr auto theta = calorically_perfect(e, cv);
constexpr auto     q = fouriers_law(theta, kappa);

/// 5. Our update equations.
constexpr auto rho_rhs = - D(rho,i) * v(i) - rho * D(v(i),i);
constexpr auto   v_rhs = - D(v(i),j) * v(j) + D(sigma(i,j),j) / rho + g(i);
constexpr auto   e_rhs = - v(i) * D(e,i) + sigma(i,j) * d(i,j) / rho - D(q(i),i) / rho;
}

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
