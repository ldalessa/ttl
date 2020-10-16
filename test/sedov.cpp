#include <ttl/ttl.hpp>
#include <fmt/core.h>

namespace {
template <typename Density, typename Energy, typename T>
constexpr auto ideal_gas(Density&& rho, Energy&& e, T&& gamma) {
  return (gamma - 1) * rho * e;
}

template <typename Pressure, typename Velocity, typename T, typename U>
constexpr auto newtonian_fluid(Pressure&& p, Velocity&& v, T&& mu, U&& muVolume) {
  ttl::Index i = 'a';
  ttl::Index j = 'b';
  ttl::Index k = 'c';

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
  ttl::Index i = 'd';
  return - D(theta,i) * conductivity;
}

/// Model parameters
constexpr ttl::Tensor    gamma = ttl::scalar("gamma");
constexpr ttl::Tensor       mu = ttl::scalar("mu");
constexpr ttl::Tensor muVolume = ttl::scalar("muVolume");
constexpr ttl::Tensor       cv = ttl::scalar("cv");
constexpr ttl::Tensor    kappa = ttl::scalar("kappa");
constexpr ttl::Tensor        g = ttl::vector("g");

/// Dependent variables
constexpr ttl::Tensor rho = ttl::scalar("rho");
constexpr ttl::Tensor   e = ttl::scalar("e");
constexpr ttl::Tensor   v = ttl::vector("v");

/// Tensor indices
constexpr ttl::Index i = 'i';
constexpr ttl::Index j = 'j';

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
// constexpr auto tsystem = ttl::make_system_of_equations(std::tie(rho, rho_rhs),
//                                                        std::tie(v, v_rhs),
//                                                        std::tie(e, e_rhs));
}

int main()
{
  // auto     d = symmetrize(D(v(i),j));
  // // auto     p = ideal_gas(rho, e, gamma);
  // auto sigma = newtonian_fluid(p, v, mu, muVolume);
  // auto theta = calorically_perfect(e, cv);
  // auto     q = fouriers_law(theta, kappa);

  // std::cout << tsystem.size() << "\n";
  // std::cout << tsystem.capacity() << "\n";
  // for (auto&& c : tsystem.constants()) {
  //   std::cout << c << "\n";
  // }
  // std::cout << ttl::dot("d") << d << "\n";
  // std::cout << ttl::dot("p") << p << "\n";
  // std::cout << ttl::dot("sigma") << sigma << "\n";
  // std::cout << ttl::dot("theta") << theta << "\n";
  // std::cout << ttl::dot("q") << q << "\n";

  // std::cout << ttl::dot("rho") << rho_rhs << "\n";
  // std::cout << ttl::dot("v") << v_rhs << "\n";
  // std::cout << ttl::dot("e") << e_rhs << "\n";
  fmt::print("rho_rhs = {:eqn}\n", rho_rhs);
  fmt::print("  v_rhs = {:eqn}\n", v_rhs);
  fmt::print("  e_rhs = {:eqn}\n", e_rhs);

  fmt::print("graph rho {{\n{:dot}}}\n", rho_rhs);
  fmt::print("graph v {{\n{:dot}}}\n", v_rhs);
  fmt::print("graph e {{\n{:dot}}}\n", e_rhs);

  return 0;
}
