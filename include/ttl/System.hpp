#pragma once

#include "Equation.hpp"
#include "ExecutableTree.hpp"
#include "Hessian.hpp"
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
  constexpr utils::box<const TensorTree> simplify(const ParseTree<M>& tree) const
  {
    TensorTreeBuilder builder = [&](const Tensor& t) {
      return is_constant(t);
    };
    return builder(tree);
  }

  constexpr void
  scalar_trees(int N, const TensorTree* tree,
               ce::dvector<utils::box<const ScalarTree>>& out) const
  {
    ScalarTreeBuilder builder(N);
    builder(tree, out);
  }

  constexpr ce::dvector<utils::box<const ScalarTree>>
  scalar_trees(int N, const TensorTree* tree) const
  {
    ce::dvector<utils::box<const ScalarTree>> out;
    ScalarTreeBuilder builder(N);
    builder(tree, out);
    return out;
  }

  constexpr auto
  scalar_trees(int N) const
  {
    TensorTreeBuilder builder = [&](const Tensor& t) {
      return is_constant(t);
    };
    ce::dvector<utils::box<const ScalarTree>> out;
    std::apply([&](auto const&... tree) {
      (scalar_trees(N, utils::box(builder(tree)), out), ...);
    }, rhs);
    return out;
  }

  constexpr void
  scalars(int N, const ScalarTree* tree, utils::set<Scalar>& out) const
  {
    if (tag_is_binary(tree->tag)) {
      scalars(N, tree->a(), out);
      scalars(N, tree->b(), out);
    }
    if (tree->tag == TENSOR) {
      out.emplace(N, tree);
    }
  }

  constexpr utils::set<Scalar>
  scalars(int N) const
  {
    utils::set<Scalar> out;
    for (auto&& tree : scalar_trees(N)) {
      scalars(N, tree, out);
    }
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
