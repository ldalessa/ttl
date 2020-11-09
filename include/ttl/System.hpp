#pragma once

#include "DynamicTree.hpp"
#include "Equation.hpp"
#include <ce/dvector.hpp>
#include <span>
#include <tuple>

namespace ttl
{
template <typename... Tree> requires(is_tree<Tree> && ...)
struct System
{
  static constexpr int M = sizeof...(Tree);
  const Tensor* lhs_[M];
  std::tuple<Tree...> rhs_;

  constexpr System(is_equation auto&&... eqns) noexcept
      : lhs_ { eqns.lhs... }
      , rhs_ { eqns.rhs... }
  {
  }

  template <typename... Tuples>
  constexpr System(Tuples&&... tuples) noexcept
      : lhs_ {std::get<0>(tuples)...}
      , rhs_ {std::get<1>(tuples)...}
  {}

  constexpr friend int size(const System&) {
    return M;
  }

  template <int i>
  constexpr friend auto rhs(const System& system) {
    return std::get<i>(system.rhs_);
  }

  constexpr decltype(auto) tensors() const {
    return std::span(lhs_);
  }

  constexpr void constants(ce::dvector<const Tensor*>& out, auto const& tree) const {
    for (auto&& node : tree) {
      if (const Tensor* t = node.tensor()) {
        if (auto i = std::find(lhs_, lhs_ + M, t); i == lhs_ + M) {
          if (auto j = std::find(out.begin(), out.end(), t); j == out.end()) {
            out.push_back(t);
          }
        }
      }
    }
  }

  constexpr auto constants() const {
    ce::dvector<const Tensor*> out;
    std::apply([&](auto&... tree) {
      (constants(out, tree), ...);
    }, rhs_);
    return out;
  }

  // template <typename Tree>
  // constexpr void hessians(utils::set<Hessian, N>& out, const Tree& tree) const
  // {
  //   constexpr int N = Tree::size();

  //   // left to right traversal to collect parent ids
  //   int parent[N];
  //   utils::stack<int> stack;
  //   for (int i = 0, e = tree.size(); i < e; ++i) {
  //     if (tree.at(i).is_binary()) {
  //       parent[stack.pop()] = i;
  //       parent[stack.pop()] = i;
  //     }
  //     stack.push(i);
  //   }
  //   parent[stack.pop()] = N;

  //   // right to left traversal to propagate dx down the tree
  //   Index dx[N + 1] = {};
  //   for (int i = N - 1; i >= 0; --i)
  //   {
  //     TaggedNode node = tree.at(i);
  //     int pid = parent[i];
  //     dx[i] = dx[pid];

  //     if (node.is(INDEX) && tree.at(pid).is(PARTIAL)) {
  //       dx[pid] = *node.index() + dx[i];
  //     }
  //     else if (const Tensor* t = node.tensor()) {
  //       if (tensor(*t)) {
  //         if (pid < N && tree.at(pid).is(BIND)) {
  //           out.emplace(*t, dx[i], *tree.at(pid).index());
  //         }
  //         else {
  //           assert(t->order() == 0);
  //           out.emplace(*t, dx[i]);
  //         }
  //       }
  //     }
  //   }
  // }

  // constexpr auto hessians() const {
  //   utils::set<Hessian, N> out;
  //   std::apply([&](auto&... tree) {
  //     (hessians(out, tree), ...);
  //   }, rhs_);
  //   return out;
  // }

  constexpr auto simplify(is_tree auto const& tree) const {
    return TensorTree(tree, constants());
  }

  constexpr auto simplify() const {
    return std::apply([&](is_tree auto const&... tree) {
      return std::tuple(simplify(tree)...);
    }, rhs_);
  }
};

template <typename... Tuples>
System(Tuples...) -> System<std::decay_t<std::tuple_element_t<1, Tuples>>...>;

template <typename... Tree> requires(is_tree<Tree> && ...)
System(Equation<Tree>...) -> System<Tree...>;

template <typename... Equations> requires(is_equation<Equations> && ...)
constexpr auto system(Equations&&... eqns) {
  return System(std::forward<Equations>(eqns)...);
}
}
