#pragma once

#include <algorithm>

namespace ttl
{
  struct TreeShape
  {
    int  node_depth = 1;
    int  node_count = 1;
    int  index_size;
    int stack_depth;

    constexpr TreeShape(int index_size, int stack_depth)
        : index_size(index_size)
        , stack_depth(stack_depth)
    {
    }

    constexpr TreeShape(int index_size, TreeShape const& a, TreeShape const& b)
        : node_depth(std::max(a.node_depth, b.node_depth) + 1)
        , node_count(a.node_count + b.node_count + 1)
        , index_size(a.index_size + b.index_size + index_size)
        , stack_depth(std::max(a.stack_depth, b.stack_depth))
    {
    }
  };
}
