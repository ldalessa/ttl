#pragma once

#include "Equation.hpp"
#include "ExecutableTree.hpp"
#include "ParseTree.hpp"
#include "Scalar.hpp"
#include "ScalarTree.hpp"
#include "TensorTree.hpp"
#include "concepts.hpp"
#include "kumi.hpp"
#include "pow.hpp"
#include "set.hpp"

namespace ttl {
  template <kumi::product_type Equations>
  struct System
  {
    Equations equations;

    constexpr System(is_equation auto... eqns)
        : equations { std::move(eqns)... }
    {
    }

    constexpr auto lhs(auto&& op) const
    {
      return equations([&](is_equation auto const&... eqns) {
        return op(eqns.lhs...);
      });
    }

    constexpr auto rhs(auto&& op) const
    {
      return equations([&](is_equation auto const&... eqns) {
        return op(eqns.rhs...);
      });
    }

    constexpr bool is_constant(const Tensor& t) const
    {
      return lhs([&](auto const& ... u) {
        return ((t != u) && ...);
      });
    }

    constexpr int n_scalar_trees(int N) const {
      return rhs([N](auto const&... tree) {
        return (0 + ... + pow(N, tree.order()));
      });
    }

    constexpr TensorTree simplify(const Tensor& lhs, is_tree auto const& tree) const
    {
      return TensorTree(lhs, tree, [&](const Tensor& t) {
        return is_constant(t);
      });
    }

    constexpr auto simplify_trees() const
    {
      return equations([&](is_equation auto const&... eqns) {
        return kumi::tuple{simplify(eqns.lhs, eqns.rhs)...};
      });
    }

    /// Returns a tuple of shapes for the trees.
    ///
    /// The passed dimensionality allows us to know how large the stack depth of
    /// tensor data is going to be.
    constexpr auto shapes(int N) const
    {
      auto trees = simplify_trees();
      return trees([N](is_tree auto const& ... trees) {
        return kumi::make_tuple(trees.shape(N)...);
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

      equations([&](is_equation auto const&... eqns) {
        (scalar_trees(N, TensorTree(eqns.lhs, eqns.rhs, constants), out), ...);
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

  System(is_equation auto... eqns)
    -> System<decltype(kumi::tuple{std::move(eqns)...})>;
}
