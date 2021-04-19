#pragma once

#include "Equation.hpp"
#include "ParseTree.hpp"
#include "TensorTree.hpp"
#include "concepts.hpp"
#include "kumi.hpp"
#include "pow.hpp"
#include "set.hpp"

namespace ttl {
  template <kumi::product_type Equations>
  struct System
  {
    /// Create a system of equations from a pack of equations.
    constexpr System(is_equation auto... eqns)
        : equations { std::move(eqns)... }
    {
    }

    /// The tuple of equations.
    ///
    /// As with all of our tuples, this can also be passed an operator which it
    /// will evaluate via an apply.
    Equations equations;

    /// Evaluate an operator for the left-hand-sides of the system.
    constexpr auto lhs(auto const& op) const -> decltype(auto)
    {
      return equations([&](is_equation auto const&... eqns) {
        return op(eqns.lhs...);
      });
    }

    /// Evaluate an operator for the right-hand-sides of the system.
    constexpr auto rhs(auto const& op) const -> decltype(auto)
    {
      return equations([&](is_equation auto const&... eqns) {
        return op(eqns.rhs...);
      });
    }

    /// Check to see if the passed tensor is a constant.
    ///
    /// Currently limited to just checking to see if the tensor appears on the
    /// left-hand-side of a pde update equation. In the future we would like
    /// this to be more sophisticated.
    constexpr bool is_constant(Tensor const& t) const
    {
      return lhs([&](auto const& ... u) {
        return ((t != u) && ...);
      });
    }

    /// Simplify a parse tree to create a tensor tree.
    ///
    /// The simpified tree is a traditional dynamically allocated tree of nodes,
    /// not an expression tree, and thus can't be leaked from the constexpr
    /// context.
    constexpr auto simplify(Tensor const& lhs, is_tree auto const& tree) const
      -> TensorTree
    {
      return TensorTree(lhs, tree, [&](const Tensor& t) {
        return is_constant(t);
      });
    }

    /// Create a tuple of simplified trees corresponding to the system.
    constexpr auto simplify_trees() const -> kumi::product_type auto
    {
      return equations([&](is_equation auto const&... eqns) {
        return kumi::make_tuple(simplify(eqns.lhs, eqns.rhs)...);
      });
    }

    /// Returns a tuple of shapes for the simplified trees.
    ///
    /// This shape depends on the dimensionality, as it requires knowledge about
    /// how many scalars are going to be associated with tensors an immediate
    /// values.
    constexpr auto shapes(int N) const -> kumi::product_type auto
    {
      auto trees = simplify_trees();
      return trees([N](is_tree auto const& ... trees) {
        return kumi::make_tuple(trees.shape(N)...);
      });
    }

    /// Create a tuple of pairs of shapes and simplified trees.
    constexpr auto simplify_trees(int N) const
    {
      return kumi::zip(shapes(N), simplify_trees());
    }
  };

  System(is_equation auto... eqns)
    -> System<decltype(kumi::tuple{std::move(eqns)...})>;
}
