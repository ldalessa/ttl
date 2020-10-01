#include <ttl/ttl.hpp>
#include <iostream>

namespace {
template <typename Density, typename Energy, typename T>
constexpr auto ideal_gas(Density&& rho, Energy&& e, T&& gamma) {
  return (gamma - 1) * rho * e;
}

template <typename Pressure, typename Velocity, typename T, typename U>
constexpr auto newtonian_fluid(Pressure&& p, Velocity&& v, T&& mu, U&& muVolume) {
  ttl::index i = ttl::idx<'a'>;
  ttl::index j = ttl::idx<'b'>;
  ttl::index k = ttl::idx<'c'>;

  auto   d = symmetrize(D(v(i),j));
  auto iso = p + muVolume * D(v(k),k);
  auto dev = 2 * mu * d - 2.0 / 3.0 * mu * D(v(k),k) * delta(i,j);
  return iso * delta(i,j) + dev;
}

template <typename Energy, typename T>
constexpr auto calorically_perfect(Energy&& e, T&& specific_heat) {
  return e / specific_heat;
}

template <typename Temperature, typename T>
constexpr auto fouriers_law(Temperature&& theta, T&& conductivity) {
  ttl::index i = ttl::idx<'c'>;
  return - D(theta,i) * conductivity;
}

/// Model parameters
constexpr ttl::tensor    gamma = ttl::scalar("gamma");
constexpr ttl::tensor       mu = ttl::scalar("mu");
constexpr ttl::tensor muVolume = ttl::scalar("muVolume");
constexpr ttl::tensor       cv = ttl::scalar("cv");
constexpr ttl::tensor    kappa = ttl::scalar("kappa");
constexpr ttl::tensor        g = ttl::vector("g");

/// Dependent variables
constexpr ttl::tensor rho = ttl::scalar("rho");
constexpr ttl::tensor   e = ttl::scalar("e");
constexpr ttl::tensor   v = ttl::vector("v");

/// Tensor indices
constexpr ttl::index i = ttl::idx<'i'>;
constexpr ttl::index j = ttl::idx<'j'>;

/// Constitutive model terms
constexpr auto     d = symmetrize(D(v(i),j));
constexpr auto     p = ideal_gas(rho, e, gamma);
constexpr auto sigma = newtonian_fluid(p, v, mu, muVolume);
constexpr auto theta = calorically_perfect(e, cv);
constexpr auto     q = fouriers_law(theta, kappa);

/// Update equations
constexpr auto rho_rhs = - D(rho,i) * v(i) - rho * D(v(i),i);
constexpr auto   v_rhs = - D(v(i),j) * v(j) + D(sigma(i,j),j) / rho + g(i);
constexpr auto   e_rhs = - v(i) * D(e,i) + sigma(i,j) * d(i,j) / rho - D(q(i),i) / rho;

/// Boilerplate
constexpr auto tsystem = ttl::make_system_of_equations(std::tie(rho, rho_rhs),
                                                       std::tie(v, v_rhs),
                                                       std::tie(e, e_rhs));
}

int main()
{
  // std::cout << tsystem.size() << "\n";
  // std::cout << tsystem.capacity() << "\n";
  // for (auto n : tsystem) {
  //   std::cout << n << "\n";
  // }
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
