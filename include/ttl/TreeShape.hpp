#pragma once

#include <raberu.hpp>
#include <algorithm>

namespace ttl
{
  namespace kw {
    using namespace rbr::literals;
    inline constexpr auto dims = "dims"_kw;
    inline constexpr auto n_immediates = "n_immediates"_kw;
    inline constexpr auto n_scalars = "n_scalars"_kw;
    inline constexpr auto n_indices = "n_indices"_kw;
    inline constexpr auto n_inner_indices = "n_inner_indices"_kw;
    inline constexpr auto n_tensor_indices = "n_tensor_indices"_kw;
    inline constexpr auto stack_depth = "stack_depth"_kw;
    inline constexpr auto n_tensor_ids = "n_tensor_ids"_kw;
  }

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

    constexpr TreeShape(rbr::concepts::option auto... params)
    {
      rbr::settings args = { params... };
      n_scalars        = args[kw::n_scalars        | 0]; // default: 0
      n_immediates     = args[kw::n_immediates     | 0]; // default: 0
      n_inner_indices  = args[kw::n_inner_indices  | 0]; // default: 0
      n_tensor_indices = args[kw::n_tensor_indices | 0]; // default: 0
      n_tensor_ids     = args[kw::n_tensor_ids     | 0]; // default: 0
      dims             = args[kw::dims];                 // required
      n_indices        = args[kw::n_indices];            // required
      stack_depth      = args[kw::stack_depth];          // required
    }

    constexpr TreeShape(TreeShape const& a, TreeShape const& b, rbr::concepts::option auto... params)
        : tree_depth(std::max(a.tree_depth, b.tree_depth) + 1)
        , n_nodes(a.n_nodes + b.n_nodes + 1)
        , n_scalars(a.n_scalars + b.n_scalars)
        , n_immediates(a.n_immediates + b.n_immediates)
        , n_inner_indices(a.n_inner_indices + b.n_inner_indices)
        , n_tensor_indices(a.n_tensor_indices + b.n_tensor_indices)
        , n_tensor_ids(a.n_tensor_ids + b.n_tensor_ids)
        , dims(a.dims)
        , n_indices(a.n_indices + b.n_indices)
        , stack_depth(std::max(a.stack_depth, b.stack_depth))
    {
      assert(a.dims == b.dims);
      rbr::settings args = { params... };
      n_inner_indices += args[kw::n_inner_indices | 0]; // default: 0
      n_indices       += args[kw::n_indices];           // required
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
