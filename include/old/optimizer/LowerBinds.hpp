#pragma once

#include "ttl/Tag.hpp"
#include "ttl/optimizer/Nodes.hpp"
#include "ttl/optimizer/Transform.hpp"

namespace ttl::optimizer
{
  struct LowerBinds : Transform<LowerBinds>
  {
    using Transform<LowerBinds>::operator();
    using Transform<LowerBinds>::visit;

    constexpr auto operator()(tags::bind, node_ptr const& bind, Index search, Index replace) const
      -> node_ptr
    {
      node_ptr child = bind->a;
      Index    index = bind->tensor_index.search_and_replace(search, replace);
      Index    inner = child->outer();
      assert(inner.size() == index.size());
      return visit(child, inner, index);
    }

    constexpr auto operator()(tags::partial, node_ptr const& partial, Index search, Index replace) const
      -> node_ptr
    {
      partial->tensor_index.search_and_replace(search, replace);
      partial->a = visit(partial->a, search, replace);
      return partial;
    }

    constexpr auto operator()(tags::tensor, node_ptr const& tensor, Index search, Index replace) const
      -> node_ptr
    {
      tensor->tensor_index.search_and_replace(search, replace);
      return tensor;
    }

    constexpr auto operator()(tags::builtin0, node_ptr const& δ, Index search, Index replace) const
      -> node_ptr
    {
      δ->tensor_index.search_and_replace(search, replace);
      return δ;
    }

    constexpr auto operator()(node_ptr node) const -> node_ptr
    {
      return visit(node, Index{}, Index{});
    }
  };
}
