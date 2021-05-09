#pragma once

#include "ttl/Equation.hpp"
#include "ttl/concepts.hpp"
#include "ttl/pow.hpp"
#include "ttl/set.hpp"
#include "ttl/optimizer/Tree.hpp"
#include <kumi.hpp>

namespace ttl
{
  template <kumi::product_type Equations>
  struct System
  {
    using is_system_tag = void;

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
    constexpr bool is_constant(TensorBase const* t) const
    {
      return lhs([&](auto const& ... u) {
        return ((t != u) && ...);
      });
    }

    /// Simplify an equation.
    constexpr auto simplify(is_equation auto const& eqn) const
      -> optimizer::Tree
    {
      return optimizer::Tree(eqn, [&](TensorBase const* tensor) -> bool {
        return is_constant(tensor);
      });
    }

    /// Create a tuple of simplified trees corresponding to the system.
    constexpr auto simplify_equations() const -> kumi::product_type auto
    {
      return equations([&](is_equation auto const&... eqns) {
        return kumi::make_tuple(simplify(eqns)...);
      });
    }

    constexpr auto simplify_equations(int N) const -> bool
    {
      auto trees = simplify_equations();
      return true;
    }

    // /// Returns a tuple of shapes for the simplified trees.
    // ///
    // /// This shape depends on the dimensionality, as it requires knowledge about
    // /// how many scalars are going to be associated with tensors an immediate
    // /// values.
    // constexpr auto shapes(int N) const -> kumi::product_type auto
    // {
    //   auto trees = simplify_trees();
    //   return trees([N](is_tree auto const& ... trees) {
    //     return kumi::make_tuple(trees.shape(N)...);
    //   });
    // }

    // /// Create a tuple of pairs of shapes and simplified trees.
    // constexpr auto simplify_trees(int N) const
    // {
    //   return kumi::zip(shapes(N), simplify_trees());
    // }
  };

  System(is_equation auto... eqns)
    -> System<decltype(kumi::tuple{std::move(eqns)...})>;
}
