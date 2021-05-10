#pragma once

#include <fmt/core.h>

namespace ttl
{
  template <typename T>
  struct dot {
    const T& tree;
    constexpr dot(const T& tree) : tree(tree) {}
  };
}

template <typename T>
struct fmt::formatter<ttl::dot<T>>
{
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  constexpr auto format(ttl::dot<T> box, auto& ctx)
  {
    return box.tree.to_dot(ctx.out());
  }
};
