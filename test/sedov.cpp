static constexpr const char USAGE[] =
 R"(sedov: run sedov
  Usage:
      sedov (-h | --help)
      sedov --version
      sedov [--constants] [--scalars] [-ptser] [--eqn <rhs>]... [--dot <rhs>]... [N]

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
      -r                 Print runtime trees
)";

#include "cm.hpp"
#include <ttl/ttl.hpp>
#include <fmt/core.h>
#include <docopt.h>
#include <cmath>

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

  // if (args["--tensors"].asBool()) {
  //   puts("tensors:");
  //   for (int i = 0; auto&& c : sedov.tensors()) {
  //     printf("%d: %s\n", i++, to_string(*c).data());
  //   }
  //   puts("");
  // }

  // if (args["--hessians"].asBool()) {
  //   puts("hessians:");
  //   for (int i = 0; auto&& c : sedov3d.hessians) {
  //     fmt::print("{}: {}\n", i++, c);
  //   }
  //   puts("");
  // }

  if (args["--constants"].asBool()) {
    puts("constants:");
    for (int i = 0; auto&& c : sedov3d.constants) {
      fmt::print("{}: {}\n", i++, c);
    }
    puts("");
  }

  // if (args["--scalars"].asBool()) {
  //   puts("scalars:");
  //   for (int n = 0; n < 8; ++n) {
  //     printf("dx in %d\n", n);
  //     for (int i = 0; int dx : sedov3d.scalars.dx(n)) {
  //       fmt::print("({},{}): {}\n", i++, dx, sedov3d.scalars[dx]);
  //     }
  //     puts("");
  //   }
  //   puts("");
  // }

  if (args["--scalars"].asBool()) {
    puts("scalars:");
    // for (int i = 0; auto&& c : sedov.scalars(N)) {
    //   fmt::print("{}: {}\n", i++, c);
    // }
    for (int n = 0; n < std::pow(2, sedov3d.dim()); ++n) {
      printf("dx: %d\n", n);
      for (int i = 0; auto&& c : sedov3d.scalars.dx(n)) {
        fmt::print("{}: {}\n", i++, c);
      }
      puts("");
    }
    puts("");
  }

  auto eqns = args["--eqn"].asStringList();
  if (ttl::utils::index_of(eqns, "rho")) {
    if (args["-p"].asBool()) {
      fmt::print("parse: rho = {}\n", rho_rhs);
    }
    if (args["-t"].asBool()) {
      fmt::print("tensor: rho = {}\n", *sedov.simplify(rho_rhs));
    }
    if (args["-s"].asBool()) {
      for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(rho_rhs))) {
        fmt::print("scalar: rho{} = {}\n", i++, *tree);
      }
    }
    // if (args["-e"].asBool()) {
    //   fmt::print("executable: rho = {}\n", rho_rhs);
    // }
    // if (args["-r"].asBool()) {
    //   fmt::print("runtime: rho = {}\n", rho_rhs);
    // }
  }

  if (ttl::utils::index_of(eqns, "v")) {
    if (args["-p"].asBool()) {
      fmt::print("parse: v = {}\n", v_rhs);
    }
    if (args["-t"].asBool()) {
      fmt::print("tensor: v = {}\n", *sedov.simplify(v_rhs));
    }
    if (args["-s"].asBool()) {
      for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(v_rhs))) {
        fmt::print("scalar: v{} = {}\n", i++, *tree);
      }
    }
    // if (args["-e"].asBool()) {
    //   fmt::print("executable: v = {}\n", v_rhs);
    // }
    // if (args["-r"].asBool()) {
    //   fmt::print("runtime: v = {}\n", v_rhs);
    // }
  }

  if (ttl::utils::index_of(eqns, "e")) {
    if (args["-p"].asBool()) {
      fmt::print("parse: e = {}\n", e_rhs);
    }
    if (args["-t"].asBool()) {
      fmt::print("tensor: e = {}\n", *sedov.simplify(e_rhs));
    }
    if (args["-s"].asBool()) {
      for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(e_rhs))) {
        fmt::print("scalar: e{} = {}\n", i++, *tree);
      }
    }
    // if (args["-e"].asBool()) {
    //   fmt::print("executable: e = {}\n", e_rhs);
    // }
    // if (args["-r"].asBool()) {
    //   fmt::print("runtime: e = {}\n", e_rhs);
    // }
  }

  auto dots = args["--dot"].asStringList();
  if (ttl::utils::contains(dots, "rho")) {
    if (args["-p"].asBool()) {
      fmt::print("graph rho_parse {{\n{}}}\n", ttl::dot(rho_rhs.root()));
    }
    if (args["-t"].asBool()) {
      fmt::print("graph rho_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(rho_rhs)));
    }
    if (args["-s"].asBool()) {
      for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(rho_rhs))) {
        fmt::print("graph rho{} {{\n{}}}\n", i++, ttl::dot(tree));
      }
    }
  }

  if (ttl::utils::contains(dots, "v")) {
    if (args["-p"].asBool()) {
      fmt::print("graph v_parse {{\n{}}}\n", ttl::dot(v_rhs.root()));
    }
    if (args["-t"].asBool()) {
      fmt::print("graph v_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(v_rhs)));
    }
    if (args["-s"].asBool()) {
      for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(v_rhs))) {
        fmt::print("graph v{} {{\n{}}}\n", i++, ttl::dot(tree));
      }
    }
  }

  if (ttl::utils::contains(dots, "e")) {
    if (args["-p"].asBool()) {
      fmt::print("graph e_parse {{\n{}}}\n", ttl::dot(e_rhs.root()));
    }
    if (args["-t"].asBool()) {
      fmt::print("graph e_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(e_rhs)));
    }
    if (args["-s"].asBool()) {
      for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(e_rhs))) {
        fmt::print("graph e{} {{\n{}}}\n", i++, ttl::dot(tree));
      }
    }
  }

  // gamma    = 1.4;       // [-]ratio of specific heats
  // cv       = 717.5;     // [J/kg.K] specific heat at constant volume
  // kappa    = 0.02545;   // [W/m.K] thermal conductivity
  // mu       = 1.9e-5;    // [Pa.s] dynamic viscosity
  // muVolume = 1e-5;      // [Pa.s] volume viscosity
  // g        = {0, 0, 0}; //

  return 0;
}
