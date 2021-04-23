// constitutive modeling for sedov
#pragma once

#include <ttl/ttl.hpp>

namespace cm
{
  template <typename Density, typename Energy, typename T>
  constexpr auto ideal_gas(Density&& ρ, Energy&& e, T&& γ)
  {
    return (γ - 1) * ρ * e;
  }

  template <typename Pressure, typename Velocity, typename T, typename U>
  constexpr auto newtonian_fluid(Pressure&& p, Velocity&& v, T&& μ, U&& μv)
  {
    using namespace ttl::literals;
    ttl::Index i = 'a';
    ttl::Index j = 'b';
    ttl::Index k = 'c';

    auto   d = symmetrize(D(v(i),j));
    auto iso = p + μv * D(v(k),k);
    auto dev = 2 * μ * d - 2_q / 3 * μ * D(v(k),k) * δ(i,j);
    return δ(i,j) * iso + dev;
  }

  template <typename Energy, typename T>
  constexpr auto calorically_perfect(Energy&& e, T&& cv)
  {
    return e / cv;
  }

  template <typename Temperature, typename T>
  constexpr auto fouriers_law(Temperature&& θ, T&& κ)
  {
    ttl::Index i = 'd';
    return - D(θ,i) * κ;
  }
}
