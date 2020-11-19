static constexpr const char USAGE[] =
 R"(sedov: run sedov
  Usage:
      sedov (-h | --help)
      sedov --version
      sedov [--constants] [--scalars] [-ptse] [--eqn <rhs>]... [--dot <rhs>]... [N]

  Options:
      -h, --help         Show this screen.
      --version          Show version information.
      --eqn <rhs>        Print an eqn for <rhs>
      --dot <rhs>        Print a dotfile for <rhs>
      --constants        Print a list of the constant scalars in the system
      --scalars          Print a list of the scalars in the system
      -p                 Print parse trees
      -t                 Print tensor treesx
      -s                 Print scalar trees
      -e                 Print executable trees
)";

#include "cm.hpp"
#include <ttl/ttl.hpp>
#include <fmt/core.h>
#include <docopt.h>
#include <vector>

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

constexpr ttl::ScalarSystem<sedov, 3> sedov3d;
}

int main(int argc, char* const argv[])
{
  std::map args = docopt::docopt(USAGE, {argv + 1, argv + argc});

  int N = args["N"] ? args["N"].asLong() : 3;

  if (args["--constants"].asBool())
  {
    puts("constants:");
    for (int i = 0; auto&& c : sedov3d.constants) {
      fmt::print("{}: {}\n", i++, c);
    }
    puts("");
  }

  if (args["--scalars"].asBool())
  {
    puts("scalars:");
    for (int j = 0, n = 0; n < ttl::utils::pow(2, sedov3d.dim()); ++n) {
      printf("dx: %d\n", n);
      for (int i = 0; auto&& c : sedov3d.scalars.dx(n)) {
        fmt::print("{} {}: {}\n", j++, i++, c);
      }
      puts("");
    }
    puts("");
  }

  auto eqns = args["--eqn"].asStringList();
  if (ttl::utils::index_of(eqns, "rho")) {
    if (args["-p"].asBool()) {
      fmt::print("parse: {} = {}\n", v, rho_rhs.to_string());
    }
    if (args["-t"].asBool()) {
      fmt::print("tensor: {}\n", sedov.simplify(rho, rho_rhs).to_string());
    }
    if (args["-s"].asBool()) {
      for (auto&& tree : sedov.scalar_trees(N, sedov.simplify(rho, rho_rhs))) {
        fmt::print("scalar: {}\n", tree.to_string());
      }
    }
    if (args["-e"].asBool()) {
      constexpr int M = sedov3d.scalars(rho);
      fmt::print("exec rho: {}\n", std::get<M>(sedov3d.executable).to_string());
    }
  }

  if (ttl::utils::index_of(eqns, "v")) {
    if (args["-p"].asBool()) {
      fmt::print("parse: {} = {}\n", v, v_rhs.to_string());
    }
    if (args["-t"].asBool()) {
      fmt::print("tensor: {}\n", sedov.simplify(v, v_rhs).to_string());
    }
    if (args["-s"].asBool()) {
      for (auto&& tree : sedov.scalar_trees(N, sedov.simplify(v, v_rhs))) {
        fmt::print("scalar: {}\n", tree.to_string());
      }
    }
    if (args["-e"].asBool()) {
      [&]<std::size_t... n>(std::index_sequence<n...>) {
        (fmt::print("exec v[{}]: {}\n", n, std::get<sedov3d.scalars(v, n)>(sedov3d.executable).to_string()), ...);
      }(std::make_index_sequence<sedov3d.dim()>());
    }
  }

  if (ttl::utils::index_of(eqns, "e")) {
    if (args["-p"].asBool()) {
      fmt::print("parse: {} = {}\n", e, e_rhs.to_string());
    }
    if (args["-t"].asBool()) {
      fmt::print("tensor: {}\n", sedov.simplify(e, e_rhs).to_string());
    }
    if (args["-s"].asBool()) {
      for (auto&& tree : sedov.scalar_trees(N, sedov.simplify(e, e_rhs))) {
        fmt::print("scalar: {}\n", tree.to_string());
      }
    }
    if (args["-e"].asBool()) {
      constexpr int M = sedov3d.scalars(e);
      fmt::print("exec e: {}\n", std::get<M>(sedov3d.executable).to_string());
    }
  }

  auto dots = args["--dot"].asStringList();
  if (ttl::utils::contains(dots, "rho")) {
    if (args["-p"].asBool()) {
      fmt::print("graph rho_parse {{\n{}}}\n", ttl::dot(rho_rhs));
    }
    if (args["-t"].asBool()) {
      fmt::print("graph rho_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(rho, rho_rhs)));
    }
    if (args["-s"].asBool()) {
      for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(rho, rho_rhs))) {
        fmt::print("graph rho{} {{\n{}}}\n", i++, ttl::dot(tree));
      }
    }
  }

  if (ttl::utils::contains(dots, "v")) {
    if (args["-p"].asBool()) {
      fmt::print("graph v_parse {{\n{}}}\n", ttl::dot(v_rhs));
    }
    if (args["-t"].asBool()) {
      fmt::print("graph v_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(v, v_rhs)));
    }
    if (args["-s"].asBool()) {
      for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(v, v_rhs))) {
        fmt::print("graph v{} {{\n{}}}\n", i++, ttl::dot(tree));
      }
    }
  }

  if (ttl::utils::contains(dots, "e")) {
    if (args["-p"].asBool()) {
      fmt::print("graph e_parse {{\n{}}}\n", ttl::dot(e_rhs));
    }
    if (args["-t"].asBool()) {
      fmt::print("graph e_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(e, e_rhs)));
    }
    if (args["-s"].asBool()) {
      for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(e, e_rhs))) {
        fmt::print("graph e{} {{\n{}}}\n", i++, ttl::dot(tree));
      }
    }
  }

  std::vector<double> now[sedov3d.n_scalars()];
  std::vector<double> next[sedov3d.n_scalars()];
  double constants[sedov3d.n_constants()];

  constants[sedov3d.constants(gamma)]    = 1.4;     // [-]ratio of specific heats
  constants[sedov3d.constants(cv)]       = 717.f;   // [J/kg.K] specific heat at constant volume
  constants[sedov3d.constants(kappa)]    = 0.02545; // [W/m.K] thermal conductivity
  constants[sedov3d.constants(mu)]       = 1.9e-5;  // [Pa.s] dynamic viscosity
  constants[sedov3d.constants(muVolume)] = 1e-5;    // [Pa.s] volume viscosity
  constants[sedov3d.constants(g, 0)]     = 0;       // no gravity
  constants[sedov3d.constants(g, 1)]     = 0;       // no gravity
  constants[sedov3d.constants(g, 2)]     = 0;       // no gravity

  sedov3d.evaluate(std::atoi(argv[1]),
                   [&](int n, int i) -> double& {
                     return next[n][i];
                   },
                   [&](int n, int i) -> double {
                     return now[n][i];
                   },
                   [&](int n) -> double {
                     return constants[n];
                   });

  return 0;
}
