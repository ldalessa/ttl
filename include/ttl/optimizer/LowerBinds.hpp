#pragma once

#include "ttl/optimizer/Nodes.hpp"

namespace ttl::optimizer
{
  struct LowerBinds
  {
    constexpr auto operator()(node_ptr node) const -> Node*
    {
      return node.visit(*this, Index{}, Index{});
    }

    constexpr auto operator()(Binary* node, Index const& search, Index const& replace) const -> Node*
    {
      node->a.visit(*this, search, replace);
      node->b.visit(*this, search, replace);
      return node;
    }

    constexpr auto operator()(Unary* node, Index const& search, Index const& replace) const -> Node*
    {
      node->a.visit(*this, search, replace);
      return node;
    }

    constexpr auto operator()(Leaf* leaf, Index const& search, Index const& replace) const -> Node*
    {
      return leaf;
    }

    constexpr auto operator()(Bind* bind, Index const& search, Index const& replace) const -> Node*
    {

      node_ptr child = bind->a;
      Index    index = bind->index.search_and_replace(search, replace);
      Index    inner = child->outer();
      assert(inner.size() == index.size());
      if (Node* parent = bind->parent) {
        parent->replace(bind, child);
      }
      else {
        child->parent = nullptr;
      }
      return child.visit(*this, inner, index);
    }

    constexpr auto operator()(Partial* partial, Index const& search, Index const& replace) const -> Node*
    {
      partial->index.search_and_replace(search, replace);
      partial->a.visit(*this, search, replace);
      return partial;
    }

    constexpr auto operator()(Tensor* tensor, Index const& search, Index const& replace) const -> Node*
    {
      tensor->index.search_and_replace(search, replace);
      return tensor;
    }

    constexpr auto operator()(Delta* δ, Index const& search, Index const& replace) const -> Node*
    {
      δ->index.search_and_replace(search, replace);
      return δ;
    }

    constexpr auto operator()(Epsilon* ε, Index const& search, Index const& replace) const -> Node*
    {
      ε->index.search_and_replace(search, replace);
      return ε;
    }
  };
}
