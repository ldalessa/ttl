#pragma once

#include "Equation.hpp"
#include "ExecutableTree.hpp"
#include "ParseTree.hpp"
#include "Scalar.hpp"
#include "ScalarTree.hpp"
#include "TensorTree.hpp"
#include "utils.hpp"
#include <tuple>

namespace ttl {
template <typename... Trees>
requires(is_tree<Trees> && ...)
struct System
{
  constexpr static int M = sizeof...(Trees);

  Tensor lhs[M];
  std::tuple<Trees...> rhs;

  constexpr System(Equation<Trees>&&... eqns)
      : lhs { eqns.lhs... }
      , rhs { eqns.rhs... }
  {
  }

  constexpr bool is_constant(const Tensor& t) const
  {
    for (const Tensor& u : lhs) {
      if (t == u) return false;
    }
    return true;
  }

  template <int M>
  constexpr TensorTree simplify(const ParseTree<M>& tree) const
  {
    return TensorTree(tree, [&](const Tensor& t) {
      return is_constant(t);
    });
  }

  constexpr int n_scalar_trees(int N) const {
    return std::apply([N](auto const&... tree) {
      return (0 + ... + utils::pow(N, tree.order()));
    }, rhs);
  }

  constexpr void
  scalar_trees(int N, const TensorTree& tree, ce::dvector<ScalarTree>& out) const
  {
    ScalarTreeBuilder builder(N);
    builder(tree, out);
  }

  constexpr ce::dvector<ScalarTree>
  scalar_trees(int N, const TensorTree& tree) const
  {
    ce::dvector<ScalarTree> out;
    ScalarTreeBuilder builder(N);
    builder(tree, out);
    return out;
  }

  constexpr auto
  scalar_trees(int N) const
  {
    auto constants = [&](const Tensor& t) {
      return is_constant(t);
    };
    ce::dvector<ScalarTree> out;
    std::apply([&](auto const&... tree) {
      (scalar_trees(N, TensorTree(tree, constants), out), ...);
    }, rhs);
    return out;
  }

  constexpr utils::set<Scalar>
  scalars(int N) const
  {
    utils::set<Scalar> out;
    for (auto&& tree : scalar_trees(N)) {
      tree.scalars(out);
    }
    return out;
  }
};

template <typename... Trees>
System(Equation<Trees>...) -> System<Trees...>;
}
