#pragma once

#include <fmt/core.h>

namespace ttl {
template <typename T>
struct dot {
  const T& tree;
  constexpr dot(const T& tree) : tree(tree) {}
};
}

template <typename T>
struct fmt::formatter<ttl::dot<T>>
{
  static constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  static constexpr auto format(ttl::dot<T> box, FormatContext& ctx)
  {
    using namespace ttl;
    int i = 0;
    auto&& out = ctx.out();
    auto op = [&](auto const* tree, auto&& self) -> int {
      if (tag_is_binary(tree->tag)) {
        int a = self(tree->a(), self);
        int b = self(tree->b(), self);
        if (tree->outer().size()) {
          format_to(out, "\tnode{}[label=\"{} ↑{}\"]\n", i, tree->tag, tree->outer());
        }
        else {
          format_to(out, "\tnode{}[label=\"{}\"]\n", i, tree->tag);
        }
        format_to(out, "\tnode{} -- node{}\n", i, a);
        format_to(out, "\tnode{} -- node{}\n", i, b);
      }
      else {
        format_to(out, "\tnode{}[label=\"{}\"]\n", i, tree->to_string());
      }
      return i++;
    };
    op(box.tree.root(), op);
    return out;
  }
};
