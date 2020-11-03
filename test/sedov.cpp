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

#include <ttl/ttl.hpp>
#include <fmt/core.h>
#include <docopt.h>

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
      fmt::print("{}: {}({},{})\n", i++, c.a, c.i, c.dx);
    }
    fmt::print("\n");
  }

  if (args["--partials"].asBool()) {
    auto partials = sedov3d.partials();
    fmt::print("partials ({}):\n", partials.size());
    for (int i = 0; auto&& p : partials) {
      fmt::print("{}: {}\n", i++, p);
    }
    fmt::print("\n");
  }

  auto eqns = args["--eqn"].asStringList();
  if (ttl::utils::index_of(eqns, "rho")) fmt::print("rho_rhs = {}\n", rho_rhs);
  if (ttl::utils::index_of(eqns, "v")) fmt::print("  v_rhs = {}\n", v_rhs);
  if (ttl::utils::index_of(eqns, "e")) fmt::print("  e_rhs = {}\n", e_rhs);

  auto dots = args["--dot"].asStringList();
  if (ttl::utils::index_of(dots, "rho")) fmt::print("graph rho {{\n{:dot}}}\n", rho_rhs);
  if (ttl::utils::index_of(dots, "v")) fmt::print("graph v {{\n{:dot}}}\n", v_rhs);
  if (ttl::utils::index_of(dots, "e")) fmt::print("graph e {{\n{:dot}}}\n", e_rhs);

  return 0;
}
