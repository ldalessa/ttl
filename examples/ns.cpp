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
  constexpr ttl::Tensor  γ = ttl::scalar("γ");
  constexpr ttl::Tensor  μ = ttl::scalar("μ");
  constexpr ttl::Tensor μv = ttl::scalar("μv");
  constexpr ttl::Tensor cv = ttl::scalar("cv");
  constexpr ttl::Tensor  κ = ttl::scalar("κ");
  constexpr ttl::Tensor  g = ttl::vector("g");

  /// Dependent variables
  constexpr ttl::Tensor ρ = ttl::scalar("ρ");
  constexpr ttl::Tensor e = ttl::scalar("e");
  constexpr ttl::Tensor v = ttl::vector("v");

  /// Tensor indices
  constexpr ttl::Index i = 'i';
  constexpr ttl::Index j = 'j';

  /// Constitutive model terms
  constexpr auto d = symmetrize(D(v(i),j));
  constexpr auto p = cm::ideal_gas(ρ, e,  γ);
  constexpr auto σ = cm::newtonian_fluid(p, v, μ, μv);
  constexpr auto θ = cm::calorically_perfect(e, cv);
  constexpr auto q = cm::fouriers_law(θ, κ);

  /// System of equations.
  constexpr auto ρ_rhs = - D(ρ,i) * v(i) - ρ * D(v(i),i);
  constexpr auto v_rhs = - D(v(i),j) * v(j) + D(σ(i,j),j) / ρ + g(i);
  constexpr auto e_rhs = - v(i) * D(e,i) + σ(i,j) * d(i,j) / ρ - D(q(i),i) / ρ;

  constexpr ttl::System navier_stokes = {
    ρ <<= ρ_rhs,
    v <<= v_rhs,
    e <<= e_rhs
  };
}
template <int N>
int run_ns(auto& args)
{
  navier_stokes.equations([](ttl::is_equation auto const&... eqn) {
    (eqn([](ttl::Tensor const* lhs, const auto& rhs) {
      fmt::print("{} = {}\n", *lhs, to_string(rhs));
    }), ...);
  });

  constexpr auto b = navier_stokes.simplify_equations(1);

  navier_stokes.simplify_equations()([](auto const&... eqn) {
    (eqn.print(stdout), ...);
  });

  navier_stokes.equations([](ttl::is_equation auto const&... eqn) {
    (eqn.dot(stdout), ...);
  });

  navier_stokes.simplify_equations()([](auto const&... eqn) {
    (eqn.dot(stdout), ...);
  });

  // constexpr ttl::ExecutableSystem<double, N, navier_stokes> navier_stokes_Nd;

  // // constexpr auto trees = sedov3d.serialized_tensor_trees;
  // // auto ser = sedov3d.serialize_tensor_trees();
  // // auto trees = sedov3d.make_executable_tensor_trees();

  // if (args["--constants"].asBool())
  // {
  //   puts("constants:");
  //   for (int i = 0; auto&& c : navier_stokes_Nd.constants) {
  //     fmt::print("{}: {}\n", i++, c);
  //   }
  //   puts("");
  // }

  // if (args["--scalars"].asBool())
  // {
  //   puts("scalars:");
  //   for (int i = 0; auto&& c : navier_stokes_Nd.scalars) {
  //     fmt::print("{}: {}\n", i++, c);
  //   }
  //   puts("");
  // }

  // auto eqns = args["--eqn"].asStringList();
  // if (std::find(eqns.begin(), eqns.end(), "ρ") != eqns.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("parse: {} = {}\n", ρ, ρ_rhs.to_string());
  //   }
  //   if (args["-t"].asBool()) {
  //     fmt::print("tensor: {}\n", ttl::TensorTree(ρ, ρ_rhs, navier_stokes).to_string());
  //   }
  //   // if (args["-e"].asBool()) {
  //   //   constexpr int M = sedov3dscalar.scalars(ρ);
  //   //   fmt::print("exec ρ: {}\n", kumi::get<M>(sedov3dscalar.executable).to_string());
  //   // }
  // }

  // if (std::find(eqns.begin(), eqns.end(), "v") != eqns.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("parse: {} = {}\n", v, v_rhs.to_string());
  //   }
  // //   if (args["-t"].asBool()) {
  // //     fmt::print("tensor: {}\n", sedov.simplify(v, v_rhs).to_string());
  // //   }
  // //   if (args["-e"].asBool()) {
  // //     [&]<std::size_t... n>(std::index_sequence<n...>) {
  // //       (fmt::print("exec v[{}]: {}\n", n, kumi::get<sedov3dscalar.scalars(v, n)>(sedov3dscalar.executable).to_string()), ...);
  // //     }(std::make_index_sequence<sedov3dscalar.dim()>());
  // //   }
  // }

  // if (std::find(eqns.begin(), eqns.end(), "e") != eqns.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("parse: {} = {}\n", e, e_rhs.to_string());
  //   }
  // //   if (args["-t"].asBool()) {
  // //     fmt::print("tensor: {}\n", sedov.simplify(e, e_rhs).to_string());
  // //   }
  // //   if (args["-e"].asBool()) {
  // //     constexpr int M = sedov3dscalar.scalars(e);
  // //     fmt::print("exec e: {}\n", kumi::get<M>(sedov3dscalar.executable).to_string());
  // //   }
  // }

  // auto dots = args["--dot"].asStringList();
  // if (std::find(dots.begin(), dots.end(), "ρ") != dots.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("graph ρ_parse {{\n{}}}\n", ttl::dot(ρ_rhs));
  //   }
  // //   if (args["-t"].asBool()) {
  // //     fmt::print("graph ρ_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(ρ, ρ_rhs)));
  // //   }
  // //   if (args["-s"].asBool()) {
  // //     for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(ρ, ρ_rhs))) {
  // //       fmt::print("graph ρ{} {{\n{}}}\n", i++, ttl::dot(tree));
  // //     }
  // //   }
  // }

  // if (std::find(dots.begin(), dots.end(), "v") != dots.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("graph v_parse {{\n{}}}\n", ttl::dot(v_rhs));
  //   }
  // //   if (args["-t"].asBool()) {
  // //     fmt::print("graph v_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(v, v_rhs)));
  // //   }
  // //   if (args["-s"].asBool()) {
  // //     for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(v, v_rhs))) {
  // //       fmt::print("graph v{} {{\n{}}}\n", i++, ttl::dot(tree));
  // //     }
  // //   }
  // }

  // if (std::find(dots.begin(), dots.end(), "e") != dots.end()) {
  //   if (args["-p"].asBool()) {
  //     fmt::print("graph e_parse {{\n{}}}\n", ttl::dot(e_rhs));
  //   }
  // //   if (args["-t"].asBool()) {
  // //     fmt::print("graph e_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(e, e_rhs)));
  // //   }
  // //   if (args["-s"].asBool()) {
  // //     for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(e, e_rhs))) {
  // //       fmt::print("graph e{} {{\n{}}}\n", i++, ttl::dot(tree));
  // //     }
  // //   }
  // }

  // const std::array constants = navier_stokes_Nd.map_constants(
  //   γ = 1.4,                        // [-]ratio of specific heats
  //   cv = 717.f,                     // [J/kg.K] specific heat at constant volume
  //   κ = 0.02545,                    // [W/m.K] thermal conductivity
  //   μ = 1.9e-5,                     // [Pa.s] dynamic viscosity
  //   μv = 1e-5,                      // [Pa.s] volume viscosity
  //   g(0) = 0,                       // no gravity
  //   g(1) = 1,                       // no gravity
  //   g(2) = 2);                      // no gravity

  // navier_stokes_Nd.evaluate(
  //   [](int id, int i) {
  //     return 0;
  //   },
  //   [&](int id) {
  //     return kumi::get<1>(constants[id]);
  //   });

  // // int n = (argc > 16) ? std::stoi(argv[1]) : 128; // args["N_POINTS"].asLong() : 0;

  // // std::vector<double, ttl::SIMDAllocator<double>> now[sedov3dscalar.n_scalars()];
  // // std::vector<double, ttl::SIMDAllocator<double>> next[sedov3dscalar.n_scalars()];
  // // for (auto& v : now) {
  // //   v.resize(n);
  // // }

  // // for (auto& v : next) {
  // //   v.resize(n);
  // // }

  // // // AUTOVECTORIZED SIMD
  // // // sedov3dscalar.evaluate(n,
  // // //                  [&](int n, int i) -> double& {
  // // //                    return next[n][i];
  // // //                  },
  // // //                  [&](int n, int i) -> const double& {
  // // //                    return now[n][i];
  // // //                  },
  // // //                  [&](int n) -> double {
  // // //                    return constants[n];
  // // //                  });


  // // // MANUAL SIMD evaluation
  // // sedov3dscalar.evaluate_simd(n,
  // //                       [&](int n, int i) -> double& {
  // //                         return next[n][i];
  // //                       },
  // //                       [&](int n, int i) -> const double& {
  // //                         return now[n][i];
  // //                       },
  // //                       [&](int n) -> double {
  // //                         return constants[n];
  // //                       });
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
