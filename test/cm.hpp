// constitutive modeling for sedov
#pragma once

#include <ttl/ttl.hpp>

namespace cm {
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
  return delta(i,j) * iso + dev;
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
}
