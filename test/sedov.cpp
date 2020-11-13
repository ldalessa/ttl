static constexpr const char USAGE[] =
 R"(sedov: run sedov
  Usage:
      sedov (-h | --help)
      sedov --version
      sedov [--hessians] [--tensors] [--constants]  [--scalars] [--eqn <rhs>]... [--dot <rhs>]...

  Options:
      -h, --help         Show this screen.
      --version          Show version information.
      --hessians         Print hessians
      --constants        Print tensor constants
      --tensors          Print tensors
      --scalars          Print non-constant scalars
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
// constexpr auto   e_rhs = - v(i) * D(e,i) + sigma(i,j) * d(i,j) / rho - D(q(i),i) / rho;
constexpr auto   e_rhs = sigma(i,j) * d(i,j);

constexpr ttl::System sedov = {
  rho = rho_rhs,
  v = v_rhs,
  e = e_rhs
};

constexpr auto sedov3d = ttl::scalar_system<sedov, 2>;
}

int main(int argc, char* const argv[])
{
  std::map args = docopt::docopt(USAGE, {argv + 1, argv + argc});

  if (args["--tensors"].asBool()) {
    puts("tensors:");
    for (int i = 0; auto&& c : sedov.tensors()) {
      printf("%d: %s\n", i++, to_string(*c).data());
    }
    puts("");
  }

  if (args["--hessians"].asBool()) {
    puts("hessians:");
    for (int i = 0; auto&& c : sedov3d.hessians) {
      fmt::print("{}: {}\n", i++, c);
    }
    puts("");
  }

  if (args["--constants"].asBool()) {
    puts("constants:");
    for (int i = 0; auto&& c : sedov3d.constants) {
      fmt::print("{}: {}\n", i++, c);
    }
    puts("");
  }

  if (args["--scalars"].asBool()) {
    puts("scalars:");
    for (int n = 0; n < 8; ++n) {
      printf("dx in %d\n", n);
      for (int i = 0; int dx : sedov3d.scalars.dx(n)) {
        fmt::print("({},{}): {}\n", i++, dx, sedov3d.scalars[dx]);
      }
      puts("");
    }
    puts("");
  }

  auto eqns = args["--eqn"].asStringList();
  if (ttl::utils::index_of(eqns, "rho")) {
    fmt::print("rho_rhs = {}\n", rho_rhs);
  }

  if (ttl::utils::index_of(eqns, "v")) {
    fmt::print("  v_rhs = {}\n", v_rhs);
  }

  if (ttl::utils::index_of(eqns, "e")) {
    fmt::print("  e_rhs = {}\n", e_rhs);
  }

  auto dots = args["--dot"].asStringList();
  if (ttl::utils::contains(dots, "rho")) {
    fmt::print("graph rho {{\n{}}}\n", ttl::dot(rhs<0>(sedov)));
    fmt::print("graph rho {{\n{}}}\n", ttl::dot(std::get<0>(sedov3d.simple)));
  }

  if (ttl::utils::contains(dots, "v")) {
    fmt::print("graph v {{\n{}}}\n", ttl::dot(rhs<1>(sedov)));
    fmt::print("graph v {{\n{}}}\n", ttl::dot(std::get<1>(sedov3d.simple)));
  }

  if (ttl::utils::contains(dots, "e")) {
    fmt::print("graph e {{\n{}}}\n", ttl::dot(rhs<2>(sedov)));
    fmt::print("graph e {{\n{}}}\n", ttl::dot(std::get<2>(sedov3d.simple)));
  }

  // gamma    = 1.4;       // [-]ratio of specific heats
  // cv       = 717.5;     // [J/kg.K] specific heat at constant volume
  // kappa    = 0.02545;   // [W/m.K] thermal conductivity
  // mu       = 1.9e-5;    // [Pa.s] dynamic viscosity
  // muVolume = 1e-5;      // [Pa.s] volume viscosity
  // g        = {0, 0, 0}; //

  // auto trees = sedov3d.make_scalar_trees();
  // fmt::print("graph e {{\n{}}}\n", std::get<2>(trees));

  auto trees = sedov3d.make_scalar_tree(std::get<2>(sedov3d.simple));
  fmt::print("graph e {{\n{}}}\n", trees);

  return 0;
}
