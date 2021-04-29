#pragma once

#include "ttl/optimizer/Nodes.hpp"

namespace ttl::optimizer
{
  struct LowerBinds
  {
    constexpr auto operator()(node_ptr node) const -> node_ptr
    {
      return node.visit(*this, Index{}, Index{});
    }

    constexpr auto operator()(Binary* node, Index const& search, Index const& replace) const -> node_ptr
    {
      node->a = node->a.visit(*this, search, replace);
      node->b = node->b.visit(*this, search, replace);
      return node_ptr(node);
    }

    constexpr auto operator()(Unary* node, Index const& search, Index const& replace) const -> node_ptr
    {
      node->a = node->a.visit(*this, search, replace);
      return node_ptr(node);
    }

    constexpr auto operator()(Leaf* leaf, Index const& search, Index const& replace) const -> node_ptr
    {
      return node_ptr(leaf);
    }

    constexpr auto operator()(Bind* bind, Index const& search, Index const& replace) const -> node_ptr
    {

      node_ptr child = bind->a;
      Index    index = bind->index.search_and_replace(search, replace);
      Index    inner = child->outer();
      assert(inner.size() == index.size());
      return child.visit(*this, inner, index);
    }

    constexpr auto operator()(Partial* partial, Index const& search, Index const& replace) const -> node_ptr
    {
      partial->index.search_and_replace(search, replace);
      partial->a = partial->a.visit(*this, search, replace);
      return node_ptr(partial);
    }

    constexpr auto operator()(Tensor* tensor, Index const& search, Index const& replace) const -> node_ptr
    {
      tensor->index.search_and_replace(search, replace);
      return node_ptr(tensor);
    }

    constexpr auto operator()(Delta* δ, Index const& search, Index const& replace) const -> node_ptr
    {
      δ->index.search_and_replace(search, replace);
      return node_ptr(δ);
    }

    constexpr auto operator()(Epsilon* ε, Index const& search, Index const& replace) const -> node_ptr
    {
      ε->index.search_and_replace(search, replace);
      return node_ptr(ε);
    }
  };
}
