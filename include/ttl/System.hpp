#pragma once

#include "Equation.hpp"
#include "ExecutableTree.hpp"
#include "ParseTree.hpp"
#include "Scalar.hpp"
#include "ScalarTree.hpp"
#include "TensorTree.hpp"
#include "lambda_tuple.hpp"
#include "pow.hpp"
#include "set.hpp"

namespace ttl {
template <int M, typename Trees>
struct System
{
  Tensor lhs[M];
  Trees  rhs;

  template <typename... Ts>
  constexpr System(Equation<Ts>&&... eqns)
      : lhs { eqns.lhs... }
      , rhs { tuple(eqns.rhs...) }
  {
  }

  constexpr bool is_constant(const Tensor& t) const
  {
    for (const Tensor& u : lhs) {
      if (t == u) return false;
    }
    return true;
  }

  constexpr int n_scalar_trees(int N) const {
    return rhs([N](auto const&... tree) {
      return (0 + ... + pow(N, tree.order()));
    });
  }

  template <int N>
  constexpr TensorTree simplify(const Tensor& lhs, const ParseTree<N>& tree) const
  {
    return TensorTree(lhs, tree, [&](const Tensor& t) {
      return is_constant(t);
    });
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

    rhs([&](auto const&... tree) {
      int i = 0;
      (scalar_trees(N, TensorTree(lhs[i++], tree, constants), out), ...);
    });

    // we need to sort the tensor trees so that they can be found properly
    std::sort(out.begin(), out.end(),
              [](const ScalarTree& a, const ScalarTree& b) {
                return a.lhs() < b.lhs();
              });

    return out;
  }

  constexpr set<Scalar>
  scalars(int N) const
  {
    set<Scalar> out;
    for (auto&& tree : scalar_trees(N)) {
      tree.scalars(out);
    }
    return out;
  }
};

template <typename... Ts>
System(Equation<Ts>&&... eqns)
  -> System<sizeof...(Ts), decltype(tuple(eqns.rhs...))>;
}
