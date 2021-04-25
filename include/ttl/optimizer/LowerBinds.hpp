#pragma once

#include "ttl/optimizer/Nodes.hpp"

namespace ttl::optimizer
{
  struct LowerBinds
  {
    constexpr void operator()(node_ptr node) const
    {
      node.visit(*this, Index{});
    }

    constexpr void operator()(Binary* node, Index outer) const
    {
      node->a.visit(*this, outer);
      node->b.visit(*this, outer);
    }

    constexpr void operator()(Unary* node, Index outer) const
    {
      node->b.visit(*this, outer);
    }

    constexpr void operator()(Leaf* leaf, Index outer) const
    {
      if (!std::is_constant_evaluated())
      puts("leaf");
    }

    constexpr void operator()(Bind* bind, Index outer) const
    {
      if (!std::is_constant_evaluated())
        puts("bind");

      auto child = bind->b;

      if (bind->parent) {
        bind->parent->replace(bind, child);
      }
      else {
        child->parent = nullptr;
      }

      child.visit(*this, outer);
    }

    constexpr void operator()(Tensor* tensor, Index outer) const
    {
      if (!std::is_constant_evaluated())
      puts("tensor");
    }

    constexpr void operator()(Partial* partial, Index outer) const
    {
      if (!std::is_constant_evaluated())
      puts("partial");
    }
  };
}
