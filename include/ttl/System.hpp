#pragma once

#include "Equation.hpp"
#include "Hessian.hpp"
#include "SimpleTree.hpp"
#include "concepts.hpp"
#include "utils.hpp"
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

  constexpr static int size() {
    return M;
  }

  template <int i>
  constexpr friend auto rhs(const System& system) {
    return std::get<i>(system.rhs_);
  }

  constexpr decltype(auto) tensors() const {
    return std::span(lhs_);
  }

  constexpr decltype(auto) trees() const {
    return rhs_;
  }

  constexpr bool is_constant(const Tensor* t) const {
    return not utils::contains(lhs_, t);
  }

  constexpr void constants(utils::set<const Tensor*>& out, is_tree auto const& tree) const {
    for (auto const& node : tree) {
      if (const Tensor* t = node.tensor()) {
        if (is_constant(t)) {
          out.emplace(t);
        }
      }
    }
  }

  constexpr auto constants() const {
    utils::set<const Tensor*> out;
    std::apply([&](auto&... tree) {
      (constants(out, tree), ...);
    }, rhs_);
    return out;
  }

  constexpr void hessians(utils::set<Hessian>& out, is_tree auto const& tree) const
  {
    for (int i = tree.size() - 1; i >= 0; --i) {
      const Node& node = tree[i];
      if (const Tensor* t = node.tensor()) {
        out.emplace(t);
      }
      if (node.tag == PARTIAL || node.tag == BIND) {
        const Node& b = tree[--i]; assert(b.tag == INDEX);
        const Node& a = tree[--i]; assert(a.tag == TENSOR);
        out.emplace(a.tensor(), *b.index());
      }
    }
  }

  constexpr static auto simplify(is_tree auto const& tree, auto const& constants) {
    return SimpleTree(tree, constants);
  }

  constexpr auto simplify(is_tree auto const& tree) const {
    return simplify(tree, constants());
  }

  constexpr auto simplify() const {
    return std::apply([&](is_tree auto const&... tree) {
      return std::tuple(simplify(tree)...);
    }, rhs_);
  }
};

template <typename... Tree> requires(is_tree<Tree> && ...)
System(Equation<Tree>...) -> System<Tree...>;

template <typename... Equations> requires(is_equation<Equations> && ...)
constexpr auto system(Equations&&... eqns) {
  return System(std::forward<Equations>(eqns)...);
}
}
