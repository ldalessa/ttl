#pragma once

#include "Equation.hpp"
#include "Hessian.hpp"
#include "TaggedTree.hpp"
#include "utils.hpp"
#include <span>
#include <tuple>

namespace ttl
{
template <typename... Rhs> requires(is_tree<Rhs> && ...)
struct System
{
  static constexpr int M = sizeof...(Rhs);
  static constexpr int N = (Rhs::n_tensors() + ... + 0);
  Tensor lhs_[M];
  std::tuple<Rhs...> rhs_;

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
  constexpr friend auto& rhs(const System& system) {
    return std::get<i>(system.rhs_);
  }

  constexpr std::optional<int> tensor(Tensor t) const {
    return utils::index_of(lhs_, t);
  }

  constexpr decltype(auto) tensors() const {
    return std::span(lhs_);
  }

  constexpr void constants(utils::set<Tensor, N>& out, auto& tree) const {
    for (int i = 0, e = size(tree); i < e; ++i) {
      if (auto* t = tree.at(i).tensor()) {
        if (not utils::index_of(lhs_, *t)) {
          out.emplace(*t);
        }
      }
    }
  }

  constexpr auto constants() const {
    utils::set<Tensor, N> out;
    std::apply([&](auto&... tree) {
      (constants(out, tree), ...);
    }, rhs_);
    return out;
  }

  template <typename Tree>
  constexpr void hessians(utils::set<Hessian, N>& out, const Tree& tree) const
  {
    constexpr int N = Tree::size();
    // bottom up to build the geometry
    int   left[N] = {};
    int  right[N] = {};
    int parent[N] = {};
    utils::stack<int> stack;
    stack.reserve(tree.depth());
    for (int i = 0, e = tree.size(); i < e; ++i) {
      if (tree.at(i).is_binary()) {
        right[i] = stack.pop();
        left[i] = stack.pop();
        parent[right[i]] = i;
        parent[left[i]] = i;
      }
      stack.push(i);
    }
    parent[stack.pop()] = N;
    assert(stack.size() == 0);

    // top-down-ish traversal to build the leaf indices
    // Index index[N + 1];
    // Index dx[N + 1];
    // index[N] = {};
    // dx[N] = {};
    // for (int i = N - 1; i >= 0; --i) {
    //   Index& idx = index[i] = index[parent[i]];
    //   Index& ddx =    dx[i] = dx[parent[i]];
    //   TaggedNode node = tree.at(i);

    //   // non-constant tensors
    //   if (const Tensor* t = node.tensor()) {
    //     if (tensor(*t)) {
    //       Index all = unique(idx + ddx);
    //       Index anon = anonymous(all);
    //       idx.search_and_replace(all, anon);
    //       ddx.search_and_replace(all, anon);
    //       out.emplace(*t, idx, idx);
    //     }
    //     continue;
    //   }

    //   if (node.is(INDEX)) {
    //     assert(right[parent[i]] == 1);
    //     continue;
    //   }

    //   else if (node.is(PARTIAL)) {
    //     ddx = *tree.at(right[i]).index() + ddx;
    //   }
    //   else if (node.is(BIND)) {
    //     idx = *tree.at(right[i]).index() + idx;
    //   }
    //   else if (node.is(PRODUCT) && tree.at(left[i]).is(INDEX)) {
    //     // delta rewrites incoming ddx
    //     Index j = *tree.at(left[i]).index();
    //     Index a = { j[0] };
    //     Index b = { j[1] };
    //     ddx.search_and_replace(b, a);
    //   }
    //   else if (node.is(PRODUCT) && tree.at(right[i]).is(INDEX)) {
    //     // delta rewrites incoming ddx
    //     Index j = *tree.at(right[i]).index();
    //     Index a = { j[0] };
    //     Index b = { j[1] };
    //     ddx.search_and_replace(b, a);
    //   }
    // }

    utils::stack<Index> index = { std::in_place, Index() };
    utils::stack<Index>    dx = { std::in_place, Index() };
    stack.push(tree.size() - 1);
    while (stack.size()) {
      int     i = stack.pop();
      Index idx = index.pop();
      Index ddx = dx.pop();
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
        ddx = *tree.at(right[i]).index() + ddx;
      }
      else if (node.is(BIND)) {
        idx = *tree.at(right[i]).index() + idx;
      }
      else if (node.is(PRODUCT) && tree.at(right[i]).is(DELTA)) {
        // delta rewrites incoming ddx
        Index j = *tree.at(right[i]).index();
        Index a = { j[0] };
        Index b = { j[1] };
        ddx.search_and_replace(b, a);
      }

      if (node.is_binary()) {
        index.push(idx);
        dx.push(ddx);
        stack.push(left[i]);

        index.push(idx);
        dx.push(ddx);
        stack.push(right[i]);
      }
    }
  }

  constexpr auto hessians() const {
    utils::set<Hessian, N> out;
    std::apply([&](auto&... tree) {
      (hessians(out, tree), ...);
    }, rhs_);
    return out;
  }

  template <typename Tree>
  //constexpr
  auto simplify(const Tree& tree) const {
    constexpr auto tags = Tree::tags;
    auto ctensors = constants();
    bool constant[Tree::size()];
    utils::stack<int> stack;
    for (int i = 0; i < std::ssize(tags); ++i) {
      if (is_binary(tags[i])) {
        int l = stack.pop();
        int r = stack.pop();
        constant[i] = constant[l] & constant[r];
      }
      else if (const Tensor *t = tree.at(i).tensor()) {
        constant[i] = utils::index_of(ctensors, *t).has_value();
      }
      else {
        constant[i] = true;
      }
      stack.push(i);
    }
    assert(stack.size() == 0);
  }

  // constexpr void simplify() const {
  //   std::apply([&](auto&... tree) {
  //     (simplify(tree), ...);
  //   }, rhs_);
  // }
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
