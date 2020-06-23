#pragma once

#include "Tree.hpp"
#include "mp/variant_utils.hpp"

namespace ttl
{
template <int M>
constexpr auto top(const Tree<M>& tree, const Node& n) {
  int i = tree.index_of(n);
  return std::visit(mp::overloaded {
      [&](Binary auto& node) {
      },
      [&](Unary auto& node) {
      },
      [&](Leaf auto& node) {

      }
    }, n);
}

template <int M>
constexpr auto simplify(const Tree<M>& tree) {
  int left[M];
  tree.visit(mp::overloaded {
      [&](int i, Binary auto&&, int a, int b) {
        assert(b == i - 1);
        left[i] = a;
        return i;
      },
      [&](int i, Unary auto&&, int a) {
        assert(a == i - 1);
        left[i] = a;
        return i;
      },
      [&](int i, Leaf auto&&) {
        left[i] = -1;
        return i;
      }});

  struct {
    const Tree<M>& tree_;
    int (&left_)[M];

    constexpr auto handle(int i) const -> int {
      return std::visit([&](const auto& node) {
        if constexpr (Binary<decltype(node)>) {
          return handle(left_[i]) + handle(i - 1);
        }
        else if constexpr (Unary<decltype(node)>) {
          return handle(i - 1);
        }
        else {
          assert(Leaf<decltype(node)>);
          return 1;
        }
      }, tree_[i]);
    }
  } eval = { tree, left };

  return eval.handle(M-1);
}
}
