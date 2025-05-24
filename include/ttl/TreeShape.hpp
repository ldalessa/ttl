#pragma once

#include <algorithm>
#include <cassert>
#include <format>

namespace ttl
{
  struct TreeShape
  {
    int       tree_depth = 1;
    int          n_nodes = 1;
    int        n_scalars = 0;
    int     n_immediates = 0;
    int  n_inner_indices = 0;
    int n_tensor_indices = 0;
    int     n_tensor_ids = 0;
    int             dims;
    int        n_indices;
    int      stack_depth;

    struct params_t 
    {
      int n_scalars{};
      int n_immediates{};
      int n_inner_indices{};
      int n_tensor_indices{};
      int n_tensor_ids{};
      int dims;
      int n_indices;
      int stack_depth;
    };

    constexpr TreeShape(params_t params)
      : n_scalars(params.n_scalars)
      , n_immediates(params.n_immediates)
      , n_inner_indices(params.n_inner_indices)
      , n_tensor_indices(params.n_tensor_indices)
      , n_tensor_ids(params.n_tensor_ids)
      , dims(params.dims)
      , n_indices(params.n_indices)
      , stack_depth(params.stack_depth)
    {
    }

    constexpr TreeShape(TreeShape const& a, TreeShape const& b, params_t params)
        : tree_depth(std::max(a.tree_depth, b.tree_depth) + 1)
        , n_nodes(a.n_nodes + b.n_nodes + 1)
        , n_scalars(a.n_scalars + b.n_scalars)
        , n_immediates(a.n_immediates + b.n_immediates)
        , n_inner_indices(a.n_inner_indices + b.n_inner_indices + params.n_inner_indices)
        , n_tensor_indices(a.n_tensor_indices + b.n_tensor_indices)
        , n_tensor_ids(a.n_tensor_ids + b.n_tensor_ids)
        , dims(a.dims)
        , n_indices(a.n_indices + b.n_indices + params.n_indices)
        , stack_depth(std::max(a.stack_depth, b.stack_depth))
    {
      assert(a.dims == b.dims);
    }
  };
}

template <>
struct std::formatter<ttl::TreeShape>
{
  static constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  static constexpr auto format(ttl::TreeShape shape, auto& ctx) {
    constexpr const char* fmt =  "tree_depth:{} n_nodes:{} n_scalars:{} n_immediates:{} n_indices:{} n_inner_indices:{} stack_depth:{}";
    return format_to(ctx.out(), fmt,
                     shape.tree_depth,
                     shape.n_nodes,
                     shape.n_scalars,
                     shape.n_immediates,
                     shape.n_indices,
                     shape.n_inner_indices,
                     shape.stack_depth);
  }
};
