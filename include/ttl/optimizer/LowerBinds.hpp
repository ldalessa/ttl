#pragma once

#include "ttl/Tag.hpp"
#include "ttl/optimizer/Nodes.hpp"

namespace ttl::optimizer
{
  struct LowerBinds
  {
    constexpr auto operator()(node_ptr node) const -> node_ptr
    {
      return visit(node, *this, Index{}, Index{});
    }

    constexpr auto operator()(tags::binary, node_ptr const& node, Index const& search, Index const& replace) const
      -> node_ptr
    {
      node->a(visit(node->a(), *this, search, replace));
      node->b(visit(node->b(), *this, search, replace));
      return node;
    }

    constexpr auto operator()(tags::unary, node_ptr const& node, Index const& search, Index const& replace) const
      -> node_ptr
    {
      node->a(visit(node->a(), *this, search, replace));
      return node;
    }

    constexpr auto operator()(tags::leaf, node_ptr const& leaf, Index const&, Index const&) const
      -> node_ptr
    {
      return leaf;
    }

    constexpr auto operator()(tags::bind, node_ptr const& bind, Index const& search, Index const& replace) const
      -> node_ptr
    {
      node_ptr child = bind->a();
      Index    index = bind->tensor_index.search_and_replace(search, replace);
      Index    inner = child->outer();
      assert(inner.size() == index.size());
      return visit(child, *this, inner, index);
    }

    constexpr auto operator()(tags::partial, node_ptr const& partial, Index const& search, Index const& replace) const
      -> node_ptr
    {
      partial->tensor_index.search_and_replace(search, replace);
      partial->a(visit(partial->a(), *this, search, replace));
      return partial;
    }

    constexpr auto operator()(tags::tensor, node_ptr const& tensor, Index const& search, Index const& replace) const
      -> node_ptr
    {
      tensor->tensor_index.search_and_replace(search, replace);
      return tensor;
    }

    constexpr auto operator()(tags::builtin0, node_ptr const& δ, Index const& search, Index const& replace) const
      -> node_ptr
    {
      δ->tensor_index.search_and_replace(search, replace);
      return δ;
    }
  };
}
