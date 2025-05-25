#include "cm.hpp"
#include <CLI/CLI.hpp>
#include <kumi/tuple.hpp>

import ttl;
import std;

namespace
{
	/// Model parameters
	constexpr ttl::Tensor γ = ttl::scalar("γ");
	constexpr ttl::Tensor μ = ttl::scalar("μ");
	constexpr ttl::Tensor μv = ttl::scalar("μv");
	constexpr ttl::Tensor cv = ttl::scalar("cv");
	constexpr ttl::Tensor κ = ttl::scalar("κ");
	constexpr ttl::Tensor g = ttl::vector("g");

	/// Dependent variables
	constexpr ttl::Tensor ρ = ttl::scalar("ρ");
	constexpr ttl::Tensor e = ttl::scalar("e");
	constexpr ttl::Tensor v = ttl::vector("v");

	/// Tensor indices
	constexpr ttl::Index i = 'i';
	constexpr ttl::Index j = 'j';

	/// Constitutive model terms
	constexpr auto d = symmetrize(D(v(i), j));
	constexpr auto p = cm::ideal_gas(ρ, e, γ);
	constexpr auto σ = cm::newtonian_fluid(p, v, μ, μv);
	constexpr auto θ = cm::calorically_perfect(e, cv);
	constexpr auto q = cm::fouriers_law(θ, κ);

	/// System of equations.
	constexpr auto ρ_rhs = -D(ρ, i) * v(i) - ρ * D(v(i), i);
	constexpr auto v_rhs = -D(v(i), j) * v(j) + D(σ(i, j), j) / ρ + g(i);
	constexpr auto e_rhs = -v(i) * D(e, i) + σ(i, j) * d(i, j) / ρ - D(q(i), i) / ρ;

	constexpr auto navier_stokes = ttl::System {
		ρ <<= ρ_rhs,
		v <<= v_rhs,
		e <<= e_rhs
	};
}

namespace options
{
	std::vector<std::string> eqns;
	std::vector<std::string> dots;
	bool print_constants = false;
	bool print_scalars = false;
	bool print_parse_trees = false;
	bool print_tensor_trees = false;
	bool print_scalar_trees = false;
	bool print_executable_trees = false;
}

template <int N>
int run_ns()
{
	static constexpr auto navier_stokes_Nd = ttl::ExecutableSystem<double, N, navier_stokes>();

	// constexpr auto trees = sedov3d.serialized_tensor_trees;
	// auto ser = sedov3d.serialize_tensor_trees();
	// auto trees = sedov3d.make_executable_tensor_trees();

	if (options::print_constants) {
		puts("constants:");
		for (int i = 0; auto&& c : navier_stokes_Nd.constants) {
			std::print("{}: {}\n", i++, c);
		}
		puts("");
	}

	if (options::print_scalars) {
		puts("scalars:");
		for (int i = 0; auto&& c : navier_stokes_Nd.scalars) {
			std::print("{}: {}\n", i++, c);
		}
		puts("");
	}

	if (std::find(options::eqns.begin(), options::eqns.end(), "ρ") != options::eqns.end()) {
		if (options::print_parse_trees) {
			std::print("parse: {} = {}\n", ρ, ρ_rhs.to_string());
		}
		if (options::print_tensor_trees) {
			std::print("tensor: {}\n", ttl::TensorTree(ρ, ρ_rhs, navier_stokes).to_string());
		}
		// if (options::print_executable_trees) {
		//   constexpr int M = sedov3dscalar.scalars(ρ);
		//   std::print("exec ρ: {}\n", kumi::get<M>(sedov3dscalar.executable).to_string());
		// }
	}

	if (std::find(options::eqns.begin(), options::eqns.end(), "v") != options::eqns.end()) {
		if (options::print_parse_trees) {
			std::print("parse: {} = {}\n", v, v_rhs.to_string());
		}
		//   if (options::print_tensor_trees) {
		//     std::print("tensor: {}\n", sedov.simplify(v, v_rhs).to_string());
		//   }
		//   if (options::print_executable_trees) {
		//     [&]<std::size_t... n>(std::index_sequence<n...>) {
		//       (std::print("exec v[{}]: {}\n", n, kumi::get<sedov3dscalar.scalars(v, n)>(sedov3dscalar.executable).to_string()), ...);
		//     }(std::make_index_sequence<sedov3dscalar.dim()>());
		//   }
	}

	if (std::find(options::eqns.begin(), options::eqns.end(), "e") != options::eqns.end()) {
		if (options::print_parse_trees) {
			std::print("parse: {} = {}\n", e, e_rhs.to_string());
		}
		//   if (options::print_tensor_trees) {
		//     std::print("tensor: {}\n", sedov.simplify(e, e_rhs).to_string());
		//   }
		//   if (options::print_executable_trees) {
		//     constexpr int M = sedov3dscalar.scalars(e);
		//     std::print("exec e: {}\n", kumi::get<M>(sedov3dscalar.executable).to_string());
		//   }
	}

	if (std::find(options::dots.begin(), options::dots.end(), "ρ") != options::dots.end()) {
		if (options::print_parse_trees) {
			std::print("graph ρ_parse {{\n{}}}\n", ttl::dot(ρ_rhs));
		}
		//   if (options::print_tensor_trees) {
		//     std::print("graph ρ_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(ρ, ρ_rhs)));
		//   }
		//   if (options::print_executable_trees) {
		//     for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(ρ, ρ_rhs))) {
		//       std::print("graph ρ{} {{\n{}}}\n", i++, ttl::dot(tree));
		//     }
		//   }
	}

	if (std::find(options::dots.begin(), options::dots.end(), "v") != options::dots.end()) {
		if (options::print_parse_trees) {
			std::print("graph v_parse {{\n{}}}\n", ttl::dot(v_rhs));
		}
		//   if (options::print_tensor_trees) {
		//     std::print("graph v_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(v, v_rhs)));
		//   }
		//   if (options::print_executable_trees) {
		//     for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(v, v_rhs))) {
		//       std::print("graph v{} {{\n{}}}\n", i++, ttl::dot(tree));
		//     }
		//   }
	}

	if (std::find(options::dots.begin(), options::dots.end(), "e") != options::dots.end()) {
		if (options::print_parse_trees) {
			std::print("graph e_parse {{\n{}}}\n", ttl::dot(e_rhs));
		}
		//   if (options::print_tensor_trees) {
		//     std::print("graph e_tensor {{\n{}}}\n", ttl::dot(sedov.simplify(e, e_rhs)));
		//   }
		//   if (options::print_executable_trees) {
		//     for (int i = 0; auto&& tree : sedov.scalar_trees(N, sedov.simplify(e, e_rhs))) {
		//       std::print("graph e{} {{\n{}}}\n", i++, ttl::dot(tree));
		//     }
		//   }
	}

	const std::array constants = navier_stokes_Nd.map_constants(
		γ = 1.4, // [-]ratio of specific heats
		cv = 717.f, // [J/kg.K] specific heat at constant volume
		κ = 0.02545, // [W/m.K] thermal conductivity
		μ = 1.9e-5, // [Pa.s] dynamic viscosity
		μv = 1e-5, // [Pa.s] volume viscosity
		g(0) = 0, // no gravity
		g(1) = 1, // no gravity
		g(2) = 2); // no gravity

	navier_stokes_Nd.evaluate(
		[](int id, int i) {
			return 0;
		},
		[&](int id) {
			return kumi::get<1>(constants[id]);
		});

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

int main(int argc, char** argv)
{
	int N;

	auto app = CLI::App();
	app.add_option("N", N, "Dimensionality");
	app.add_option("--eqns", options::eqns, "Equations to print.");
	app.add_option("--dot", options::dots, "Equations to print as dot graphs");
	app.add_option("--constants", options::print_constants, "Print a list of the constants in the system");
	app.add_option("--scalars", options::print_scalars, "Print a list of the scalars in the system");
	app.add_option("-p", options::print_parse_trees, "Print the parse trees");
	app.add_option("-t", options::print_tensor_trees, "Print the tensor trees");
	app.add_option("-s", options::print_scalar_trees, "Print the scalar trees");
	app.add_option("-e", options::print_executable_trees, "Print the executable trees");
	app.parse(argc, app.ensure_utf8(argv));

	switch (N) {
	case 1:
		return run_ns<1>();
	case 2:
		return run_ns<2>();
	case 3:
		return run_ns<3>();
	}

	std::print("navier stokes only supports N=1,2,3 ({})\n", N);
	return 0;
}
