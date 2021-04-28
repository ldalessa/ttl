#pragma once

#include "ttl/optimizer/Nodes.hpp"

namespace ttl::optimizer
{
  struct ConstProp
  {
    constexpr auto operator()(node_ptr node) const -> Node*
    {
      return node.visit(*this);
    }

    constexpr auto operator()(Binary* node) const -> Node*
    {
      node->a.visit(*this);
      node->b.visit(*this);
      return node;
    }

    constexpr auto operator()(Unary* node) const -> Node*
    {
      node->a.visit(*this);
      return node;
    }

    constexpr auto operator()(Bind* bind) const -> Node*
    {
      if (std::is_constant_evaluated()) {
        assert(false);
      }
      return nullptr;
    }

    constexpr auto operator()(Leaf* leaf) const -> Node*
    {
      return leaf;
    }
  };
}
