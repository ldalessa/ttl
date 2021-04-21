static constexpr const char USAGE[] =
 R"(sedov: run sedov
  Usage:
      sedov (-h | --help)
      sedov --version
      sedov N [--constants] [--scalars] [-ptse] [--eqn <rhs>]... [--dot <rhs>]...

  Options:
      -h, --help         Show this screen.
      --version          Show version information.
      --eqn <rhs>        Print an eqn for <rhs>
      --dot <rhs>        Print a dotfile for <rhs>
      --constants        Print a list of the constant scalars in the system
      --scalars          Print a list of the scalars in the system
      -p                 Print parse trees
      -t                 Print tensor trees
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
  constexpr auto     p = cm::ideal_gas(rho, e,  gamma);
  constexpr auto sigma = cm::newtonian_fluid(p, v, mu, muVolume);
  constexpr auto theta = cm::calorically_perfect(e, cv);
  constexpr auto     q = cm::fouriers_law(theta, kappa);

  /// System of equations.
  constexpr auto rho_rhs = - D(rho,i) * v(i) - rho * D(v(i),i);
  constexpr auto   v_rhs = - D(v(i),j) * v(j) + D(sigma(i,j),j) / rho + g(i);
  constexpr auto   e_rhs = - v(i) * D(e,i) + sigma(i,j) * d(i,j) / rho - D(q(i),i) / rho;

  constexpr ttl::System navier_stokes = {
    rho = rho_rhs,
    v = v_rhs,
    e = e_rhs
  };
}

template <int N>
int run_ns(auto const& args)
{
  constexpr ttl::ExecutableSystem<double, N, navier_stokes> navier_stokes_Nd;

  // auto trees = sedov.simplify_trees();
  // trees([](auto const&... tree) {
  //   (fmt::print("\t{}\n", tree.to_string()), ...);
  //   (fmt::print("graph rho {{\n{}}}\n", ttl::dot(tree)), ...);
  // });

  constexpr auto shapes = navier_stokes.shapes(N);
  shapes([](auto const&... shape) {
    (fmt::print("{}\n", shape), ...);
  });

  // constexpr auto trees = sedov3d.serialized_tensor_trees;
  // auto ser = sedov3d.serialize_tensor_trees();
  // auto trees = sedov3d.make_executable_tensor_trees();

  // if (args["--constants"].asBool())
  // {
  //   puts("constants:");
  //   for (int i = 0; auto&& c : sedov3dscalar.constants) {
  //     fmt::print("{}: {}\n", i++, c);
  //   }
  //   puts("");
  // }

  // if (args["--scalars"].asBool())
  // {
  //   puts("scalars:");
  //   for (int j = 0, n = 0; n < ttl::pow(2, sedov3dscalar.dim()); ++n) {
  //     printf("dx: %d\n", n);
  //     for (int i = 0; auto&& c : sedov3dscalar.scalars.dx(n)) {
  //       fmt::print("{} {}: {} {}\n", j++, i++, c, c.direction);
  //     }
  //     puts("");
  //   }
  //   puts("");
  // }

  // auto eqns = args["--eqn"].asStringList();
  // if (std::find(eqns.begin(), eqns.end(), "rho") != eqns.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("parse: {} = {}\n", rho, rho_rhs.to_string());
  //   }
  //   if (args["-t"].asBool()) {
  //     fmt::print("tensor: {}\n", sedov.simplify(rho, rho_rhs).to_string());
  //   }
  //   if (args["-s"].asBool()) {
  //     for (auto&& tree : sedov.scalar_trees(N, sedov.simplify(rho, rho_rhs))) {
  //       fmt::print("scalar: {}\n", tree.to_string());
  //     }
  //   }
  //   if (args["-e"].asBool()) {
  //     constexpr int M = sedov3dscalar.scalars(rho);
  //     fmt::print("exec rho: {}\n", kumi::get<M>(sedov3dscalar.executable).to_string());
  //   }
  // }

  // if (std::find(eqns.begin(), eqns.end(), "v") != eqns.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("parse: {} = {}\n", v, v_rhs.to_string());
  //   }
  //   if (args["-t"].asBool()) {
  //     fmt::print("tensor: {}\n", sedov.simplify(v, v_rhs).to_string());
  //   }
  //   if (args["-s"].asBool()) {
  //     for (auto&& tree : sedov.scalar_trees(N, sedov.simplify(v, v_rhs))) {
  //       fmt::print("scalar: {}\n", tree.to_string());
  //     }
  //   }
  //   if (args["-e"].asBool()) {
  //     [&]<std::size_t... n>(std::index_sequence<n...>) {
  //       (fmt::print("exec v[{}]: {}\n", n, kumi::get<sedov3dscalar.scalars(v, n)>(sedov3dscalar.executable).to_string()), ...);
  //     }(std::make_index_sequence<sedov3dscalar.dim()>());
  //   }
  // }

  // if (std::find(eqns.begin(), eqns.end(), "e") != eqns.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("parse: {} = {}\n", e, e_rhs.to_string());
  //   }
  //   if (args["-t"].asBool()) {
  //     fmt::print("tensor: {}\n", sedov.simplify(e, e_rhs).to_string());
  //   }
  //   if (args["-s"].asBool()) {
  //     for (auto&& tree : sedov.scalar_trees(N, sedov.simplify(e, e_rhs))) {
  //       fmt::print("scalar: {}\n", tree.to_string());
  //     }
  //   }
  //   if (args["-e"].asBool()) {
  //     constexpr int M = sedov3dscalar.scalars(e);
  //     fmt::print("exec e: {}\n", kumi::get<M>(sedov3dscalar.executable).to_string());
  //   }
  // }

  // auto dots = args["--dot"].asStringList();
  // if (std::find(dots.begin(), dots.end(), "rho") != dots.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("graph rho_parse {{\n{}}}\n", ttl::dot(rho_rhs));
  //   }
  //   if (args["-t"].asBool()) {
  //     fmt::print("graph rho_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(rho, rho_rhs)));
  //   }
  //   if (args["-s"].asBool()) {
  //     for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(rho, rho_rhs))) {
  //       fmt::print("graph rho{} {{\n{}}}\n", i++, ttl::dot(tree));
  //     }
  //   }
  // }

  // if (std::find(dots.begin(), dots.end(), "v") != dots.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("graph v_parse {{\n{}}}\n", ttl::dot(v_rhs));
  //   }
  //   if (args["-t"].asBool()) {
  //     fmt::print("graph v_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(v, v_rhs)));
  //   }
  //   if (args["-s"].asBool()) {
  //     for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(v, v_rhs))) {
  //       fmt::print("graph v{} {{\n{}}}\n", i++, ttl::dot(tree));
  //     }
  //   }
  // }

  // if (std::find(dots.begin(), dots.end(), "e") != dots.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("graph e_parse {{\n{}}}\n", ttl::dot(e_rhs));
  //   }
  //   if (args["-t"].asBool()) {
  //     fmt::print("graph e_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(e, e_rhs)));
  //   }
  //   if (args["-s"].asBool()) {
  //     for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(e, e_rhs))) {
  //       fmt::print("graph e{} {{\n{}}}\n", i++, ttl::dot(tree));
  //     }
  //   }
  // }

  // return 0;

  // auto trees = navier_stokes_Nd.make_executable_trees();
    // trees([](auto const&... tree) {
    //   (tree.evaluate([](int id, int i) { return 0; },
    //                  [](int id)        { return 0; }), ...);
    // });


  navier_stokes_Nd.evaluate([](int id, int i) { return 0; },
                            [](int id) { return 0; });

  // double constants[sedov3d.n_constants()];
  // constants[sedov3d.constants(model::gamma)] = 1.4;     // [-]ratio of specific heats
  // constants[sedov3d.constants(cv)]       = 717.f;   // [J/kg.K] specific heat at constant volume
  // constants[sedov3d.constants(kappa)]    = 0.02545; // [W/m.K] thermal conductivity
  // constants[sedov3d.constants(mu)]       = 1.9e-5;  // [Pa.s] dynamic viscosity
  // constants[sedov3d.constants(muVolume)] = 1e-5;    // [Pa.s] volume viscosity
  // constants[sedov3d.constants(g, 0)]     = 0;       // no gravity
  // constants[sedov3d.constants(g, 1)]     = 0;       // no gravity
  // constants[sedov3d.constants(g, 2)]     = 0;       // no gravity

  // int n = (argc > 16) ? std::stoi(argv[1]) : 128; // args["N_POINTS"].asLong() : 0;

  // std::vector<double, ttl::SIMDAllocator<double>> now[sedov3dscalar.n_scalars()];
  // std::vector<double, ttl::SIMDAllocator<double>> next[sedov3dscalar.n_scalars()];
  // for (auto& v : now) {
  //   v.resize(n);
  // }

  // for (auto& v : next) {
  //   v.resize(n);
  // }

  // // AUTOVECTORIZED SIMD
  // // sedov3dscalar.evaluate(n,
  // //                  [&](int n, int i) -> double& {
  // //                    return next[n][i];
  // //                  },
  // //                  [&](int n, int i) -> const double& {
  // //                    return now[n][i];
  // //                  },
  // //                  [&](int n) -> double {
  // //                    return constants[n];
  // //                  });


  // // MANUAL SIMD evaluation
  // sedov3dscalar.evaluate_simd(n,
  //                       [&](int n, int i) -> double& {
  //                         return next[n][i];
  //                       },
  //                       [&](int n, int i) -> const double& {
  //                         return now[n][i];
  //                       },
  //                       [&](int n) -> double {
  //                         return constants[n];
  //                       });
  return 0;
}

int main(int argc, char* const argv[])
{
  std::map args = docopt::docopt(USAGE, {argv + 1, argv + argc});

  switch (args["N"].asLong())
  {
   case 1: return run_ns<1>(args);
   case 2: return run_ns<2>(args);
   case 3: return run_ns<3>(args);
  }

  fmt::print("navier stokes only supports N=1,2,3 ({})\n", args["N"].asLong());
  return 0;
}
