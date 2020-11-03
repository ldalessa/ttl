#pragma once

#include "Equation.hpp"
#include "Hessian.hpp"
#include "TaggedTree.hpp"
#include "utils.hpp"
#include <array>
#include <tuple>

namespace ttl
{
template <typename... Rhs> requires(is_tree<Rhs> && ...)
struct System
{
  static constexpr int M = sizeof...(Rhs);
  static constexpr int N = (Rhs::n_tensors() + ... + 0);
  std::array<Tensor, M> lhs;
  std::tuple<Rhs...> rhs;

  constexpr System(is_equation auto&&... eqns) noexcept
      : lhs { eqns.lhs... }
      , rhs { eqns.rhs... }
  {
  }

  template <typename... Tuples>
  constexpr System(Tuples&&... tuples) noexcept
      : lhs {std::get<0>(tuples)...}
      , rhs {std::get<1>(tuples)...}
  {}

  constexpr std::optional<int> tensor(Tensor t) const {
    return utils::index_of(lhs, t);
  }

  constexpr decltype(auto) tensors() const {
    return lhs;
  }

  constexpr void constants(utils::set<Tensor, N>& out, auto& tree) const {
    for (int i = 0, e = size(tree); i < e; ++i) {
      if (auto* t = tree.at(i).tensor()) {
        if (not utils::index_of(lhs, *t)) {
          out.emplace(*t);
        }
      }
    }
  }

  constexpr auto constants() const {
    utils::set<Tensor, N> out;
    std::apply([&](auto&... tree) {
      (constants(out, tree), ...);
    }, rhs);
    return out;
  }

  constexpr void hessians(utils::set<Hessian, N>& out, auto& tree) const {
    constexpr auto geometry = std::decay_t<decltype(tree)>::geometry();
    utils::stack<Index> index = { std::in_place, Index() };
    utils::stack<Index>    dx = { std::in_place, Index() };
    utils::stack<int>   stack = { std::in_place, size(tree) - 1 };

    while (stack.size()) {
      int     i = stack.pop();
      Index idx = index.pop();
      Index ddx = dx.pop();
      int  left = geometry.left[i];
      int right = geometry.right[i];
      auto node = tree.at(i);

      if (const Tensor* t = node.tensor()) {
        if (tensor(*t)) {
          Index all = unique(idx + ddx);
          Index anon;
          for (int i = 0; i < all.size(); ++i) {
            anon.push_back(char('0' + i));
          }
          idx.search_and_replace(all, anon);
          ddx.search_and_replace(all, anon);
          out.emplace(*t, idx, ddx);
        }
      }
      else if (node.is(PARTIAL)) {
        ddx = *tree.at(right).index() + ddx;
      }
      else if (node.is(BIND)) {
        idx = *tree.at(right).index() + idx;
      }
      else if (node.is(PRODUCT) && tree.at(left).is(INDEX)) {
        // delta rewrites incoming ddx
        Index j = *tree.at(left).index();
        Index a = { j[0] };
        Index b = { j[1] };
        ddx.search_and_replace(b, a);
      }
      else if (node.is(PRODUCT) && tree.at(right).is(INDEX)) {
        // delta rewrites incoming ddx
        Index j = *tree.at(right).index();
        Index a = { j[0] };
        Index b = { j[1] };
        ddx.search_and_replace(b, a);
      }

      if (node.is_binary()) {
        index.push(idx);
        dx.push(ddx);
        stack.push(left);

        index.push(idx);
        dx.push(ddx);
        stack.push(right);
      }
    }
  }

  constexpr auto hessians() const {
    utils::set<Hessian, N> out;
    std::apply([&](auto&... tree) {
      (hessians(out, tree), ...);
    }, rhs);
    return out;
  }
};

template <typename... Tuples>
System(Tuples...) -> System<std::decay_t<std::tuple_element_t<1, Tuples>>...>;

template <typename... Rhs> requires(is_tree<Rhs> && ...)
System(Equation<Rhs>...) -> System<Rhs...>;

template <typename... Equations> requires(is_equation<Equations> && ...)
constexpr auto system(Equations&&... eqns) {
  return System(std::forward<Equations>(eqns)...);
}
}
