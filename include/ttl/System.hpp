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

  constexpr std::array<std::string_view, M> tensors() const {
    std::array<std::string_view, M> out;
    for (int i = 0; auto&& t : lhs) {
      out[i++] = t.id();
    }
    return out;
  }

  constexpr auto constants() const {
    constexpr int M = (Rhs::M + ... + 0);
    ce::cvector<std::string_view, M> out;

    auto search = [&](auto& tree) {
      for (int i = 0; i < tree.M; ++i) {
        if (auto* t = tree.at(i).tensor()) {
          if (not utils::index_of(lhs, *t)) {
            if (not utils::index_of(out, t->id())) {
              out.push_back(t->id());
            }
          }
        }
      }
    };

    std::apply([&](auto&... rhs) {
      (search(rhs), ...);
    }, rhs);

    return out;
  }

  constexpr auto hessians() const {
    constexpr int M = (Rhs::M + ... + 0);
    ce::cvector<Hessian, M> out;
    auto search = [&](auto& tree) {
      constexpr auto geometry = std::decay_t<decltype(tree)>::geometry();
      ce::cvector<Index, geometry.depth + 1> index = { std::in_place, Index() };
      ce::cvector<Index, geometry.depth + 1> dx = { std::in_place, Index() };
      ce::cvector<int, geometry.depth> stack = { std::in_place, tree.size() - 1 };
      while (stack.size()) {
        int     i = stack.pop_back();
        Index idx = index.pop_back();
        Index ddx = dx.pop_back();
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
            Hessian h(*t, idx, ddx);
            if (!utils::index_of(out, h)) {
              out.emplace_back(*t, idx, ddx);
            }
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
          index.push_back(idx);
          dx.push_back(ddx);
          stack.push_back(left);

          index.push_back(idx);
          dx.push_back(ddx);
          stack.push_back(right);
        }
      }
    };

    std::apply([&](auto&... rhs) {
      (search(rhs), ...);
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
