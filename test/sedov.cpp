static constexpr const char USAGE[] =
 R"(sedov: run sedov
  Usage:
      sedov (-h | --help)
      sedov --version
      sedov [--hessians] [--tensors] [--constants]  [--partials] [--eqn <rhs>]... [--dot <rhs>]...

  Options:
      -h, --help         Show this screen.
      --version          Show version information.
      --hessians         Print hessians
      --constants        Print constants
      --tensors          Print tensors
      --partials         Print partials
      --eqn <rhs>        Print an eqn for <rhs>
      --dot <rhs>        Print a dotfile for <rhs>
)";

#include "cm.hpp"
#include <ttl/ttl.hpp>
#include <fmt/core.h>
#include <docopt.h>

namespace {
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
constexpr auto     p = cm::ideal_gas(rho, e, gamma);
constexpr auto sigma = cm::newtonian_fluid(p, v, mu, muVolume);
constexpr auto theta = cm::calorically_perfect(e, cv);
constexpr auto     q = cm::fouriers_law(theta, kappa);

/// System of equations.
constexpr auto rho_rhs = - D(rho,i) * v(i) - rho * D(v(i),i);
constexpr auto   v_rhs = - D(v(i),j) * v(j) + D(sigma(i,j),j) / rho + g(i);
constexpr auto   e_rhs = - v(i) * D(e,i) + sigma(i,j) * d(i,j) / rho - D(q(i),i) / rho;

constexpr ttl::System sedov = {
  rho = rho_rhs,
  v = v_rhs,
  e = e_rhs
};

constexpr auto sedov3d = ttl::scalar_system<sedov, 3>;
}

int main(int argc, char* const argv[])
{
  std::map args = docopt::docopt(USAGE, {argv + 1, argv + argc});

  if (args["--tensors"].asBool()) {
    auto t = sedov.tensors();
    fmt::print("tensors ({}):\n", t.size());
    for (int i = 0; auto&& c : t) {
      fmt::print("{}: {}\n", i++, c);
    }
    fmt::print("\n");
  }

  if (args["--constants"].asBool()) {
    auto c = sedov.constants();
    fmt::print("constants ({}):\n", c.size());
    for (int i = 0; auto&& c : c.sort()) {
      fmt::print("{}: {}\n", i++, c);
    }
    fmt::print("\n");
  }

  if (args["--hessians"].asBool()) {
    auto h = sedov.hessians();
    fmt::print("hessians (capacity {}):\n", h.capacity());
    for (int i = 0; auto&& c : h.sort()) {
      fmt::print("{}: {}({},{})\n", i++, c.tensor(), c.index(), c.partial());
    }
    fmt::print("\n");
  }

  if (args["--partials"].asBool()) {
    auto partials = sedov3d.partials();
    fmt::print("partials ({}):\n", partials.size());
    for (int n = 0; n < 8; ++n) {
      fmt::print("dx in {}\n", n);
      for (int i = 0; int dx : partials.dx(n)) {
        fmt::print("({},{}): {}\n", i++, dx, partials[dx]);
      }
      fmt::print("\n");
    }
    fmt::print("\n");
  }

  auto eqns = args["--eqn"].asStringList();
  if (ttl::utils::index_of(eqns, "rho")) fmt::print("rho_rhs = {}\n", rho_rhs);
  if (ttl::utils::index_of(eqns, "v")) fmt::print("  v_rhs = {}\n", v_rhs);
  if (ttl::utils::index_of(eqns, "e")) fmt::print("  e_rhs = {}\n", e_rhs);

  auto dots = args["--dot"].asStringList();
  if (ttl::utils::index_of(dots, "rho")) {
    fmt::print("graph rho {{\n{:dot}}}\n", rhs<0>(sedov));
    fmt::print("graph rho2 {{\n{:dot}}}\n", rhs<0>(sedov3d));
  }

  if (ttl::utils::index_of(dots, "v")) {
    fmt::print("graph v {{\n{:dot}}}\n", rhs<1>(sedov));
    fmt::print("graph v2 {{\n{:dot}}}\n", rhs<1>(sedov3d));
  }

  if (ttl::utils::index_of(dots, "e")) {
    fmt::print("graph e {{\n{:dot}}}\n", rhs<2>(sedov));
    fmt::print("graph e2 {{\n{:dot}}}\n", rhs<2>(sedov3d));
  }

  // gamma    = 1.4;       // [-]ratio of specific heats
  // cv       = 717.5;     // [J/kg.K] specific heat at constant volume
  // kappa    = 0.02545;   // [W/m.K] thermal conductivity
  // mu       = 1.9e-5;    // [Pa.s] dynamic viscosity
  // muVolume = 1e-5;      // [Pa.s] volume viscosity
  // g        = {0, 0, 0}; //

  return 0;
}
