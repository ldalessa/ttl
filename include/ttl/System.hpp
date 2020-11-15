#pragma once

#include "Equation.hpp"
#include "Hessian.hpp"
#include "ParseTree.hpp"
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
  constexpr const TensorTree* simplify(const ParseTree<M>& tree) const
  {
    TreeBuilder builder = [&](const Tensor& t) {
      return is_constant(t);
    };
    return builder.map(tree.root());
  }

  constexpr void
  scalar_trees(int N, const TensorTree* tree, utils::set<const ScalarTree*>& out) const
  {
    ScalarTreeBuilder builder(N);
    builder(out, tree);
  }

  constexpr utils::set<const ScalarTree*>
  scalar_trees(int N, const TensorTree* tree) const
  {
    utils::set<const ScalarTree*> out;
    ScalarTreeBuilder builder(N);
    builder(out, tree);
    return out;
  }

  constexpr auto
  scalar_trees(int N) const
  {
    TreeBuilder builder = [&](const Tensor& t) {
      return is_constant(t);
    };
    utils::set<const ScalarTree*> out;
    std::apply([&](auto const&... tree) {
      (scalar_trees(N, builder.map(tree.root()), out), ...);
    }, rhs);
    return out;
  }

  constexpr void hessians(utils::set<Hessian>& out, const TensorTree* tree) const {
    if (tree->tag == TENSOR) {
      out.emplace(tree->tensor, tree->index);
    }
    if (tag_is_binary(tree->tag)) {
      hessians(out, tree->a());
      hessians(out, tree->b());
    }
  }
};

template <typename... Trees>
System(Equation<Trees>...) -> System<Trees...>;
}
