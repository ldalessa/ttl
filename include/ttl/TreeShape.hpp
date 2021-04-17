#pragma once

#include <raberu.hpp>
#include <algorithm>

namespace ttl
{
  namespace kw {
    using namespace rbr::literals;
    inline constexpr auto n_immediates = "n_immediates"_kw;
    inline constexpr auto n_scalars = "n_scalars"_kw;
  }

  struct TreeShape
  {
    int   tree_depth = 1;
    int      n_nodes = 1;
    int    n_scalars = 0;
    int n_immediates = 0;
    int    n_indices;
    int  stack_depth;

    constexpr TreeShape(int n_indices, int stack_depth, rbr::keyword_parameter auto param)
      requires(rbr::match<decltype(param)>::with(kw::n_immediates | kw::n_scalars))
      : n_indices(n_indices)
      , stack_depth(stack_depth)
    {
      rbr::settings args = { param };
      n_scalars = args[kw::n_scalars | 0];
      n_immediates = args[kw::n_immediates | 0];
    }

    constexpr TreeShape(int n_indices, TreeShape const& a, TreeShape const& b)
        : tree_depth(std::max(a.tree_depth, b.tree_depth) + 1)
        , n_nodes(a.n_nodes + b.n_nodes + 1)
        , n_scalars(a.n_scalars + b.n_scalars)
        , n_immediates(a.n_immediates + b.n_immediates)
        , n_indices(a.n_indices + b.n_indices + n_indices)
        , stack_depth(std::max(a.stack_depth, b.stack_depth))
    {
    }
  };
}

#include <fmt/format.h>

template <>
struct fmt::formatter<ttl::TreeShape>
{
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  constexpr auto format(ttl::TreeShape shape, auto& ctx) {
    constexpr const char* fmt =  "tree_depth:{} n_nodes:{} n_scalars:{} n_immediates:{} n_indices:{} stack_depth:{}";
    return format_to(ctx.out(), fmt, shape.tree_depth, shape.n_nodes, shape.n_scalars, shape.n_immediates, shape.n_indices, shape.stack_depth);
  }
};
