#pragma once

#include "utils.hpp"
#include <fmt/core.h>

namespace ttl {
template <typename T>
struct dot {
  T* tree;
  constexpr dot(T* tree) : tree(tree) {}
};

template <template <typename> typename Box, typename T>
dot(const Box<T>&) -> dot<T>;
}

template <typename T>
struct fmt::formatter<ttl::dot<T>>
{
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(ttl::dot<T> box, FormatContext& ctx)
  {
    using namespace ttl;
    int i = 0;
    auto&& out = ctx.out();
    auto op = [&](auto const* tree, auto&& self) -> int {
      if (tag_is_binary(tree->tag)) {
        int a = self(tree->a(), self);
        int b = self(tree->b(), self);
        format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, tree->tag, tree->outer());
        format_to(out, "\tnode{} -- node{}\n", i, a);
        format_to(out, "\tnode{} -- node{}\n", i, b);
      }
      else {
        format_to(out, "\tnode{}[label=\"{}\"]\n", i, tree->to_string());
      }
      return i++;
    };
    op(box.tree, op);
    return out;
  }
};

// This works for RPN trees that can be processed bottom up, but not all trees.
//
// template <typename T>
// struct fmt::formatter<ttl::dot<T>> {
//   constexpr auto parse(format_parse_context& ctx) {
//     return ctx.begin();
//   }

//   template <typename FormatContext>
//   constexpr auto format(ttl::dot<T> box, FormatContext& ctx)
//   {
//     ttl::utils::stack<int> stack;
//     for (int i = 0; auto&& node : box.tree) {
//       if (node.binary()) {
//         format_to(ctx.out(), "\tnode{}[label=\"{} {}\"]\n", i, node, *node.index());
//         format_to(ctx.out(), "\tnode{} -- node{}\n", i, stack.pop());
//         format_to(ctx.out(), "\tnode{} -- node{}\n", i, stack.pop());
//       }
//       else {
//         format_to(ctx.out(), "\tnode{}[label=\"{}\"]\n", i, node);
//       }
//       stack.push(i++);
//     }
//     return ctx.out();
//   }
// };
