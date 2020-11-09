#pragma once

#include "concepts.hpp"

namespace ttl {
template <typename T> requires(is_tree<std::decay_t<T>>)
struct dot {
  const T& tree;
  constexpr dot(const T& tree) : tree(tree) {}
};
}

template <typename T> requires(ttl::is_tree<std::decay_t<T>>)
struct fmt::formatter<ttl::dot<T>> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(ttl::dot<T> tree, FormatContext& ctx)
  {
    ttl::utils::stack<int> stack;
    for (int i = 0; auto&& node : tree.tree) {
      if (node.binary()) {
        format_to(ctx.out(), "\tnode{}[label=\"{} {}\"]\n", i, node, *node.index());
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, stack.pop());
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, stack.pop());
      }
      else {
        format_to(ctx.out(), "\tnode{}[label=\"{}\"]\n", i, node);
      }
      stack.push(i++);
    }
    return ctx.out();
  }
};
