#pragma once

#include "ttl/Tag.hpp"
#include "ttl/optimizer/Nodes.hpp"
#include "ttl/optimizer/Transform.hpp"

namespace ttl::optimizer
{
  struct Simplify : Transform<Simplify>
  {
    using Transform<Simplify>::operator();
    using Transform<Simplify>::visit;

    constexpr auto operator()(tags::sum, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a);
      node_ptr b = visit(node->b);

      // (-a) + b -> b - a
      if (a->is_negate()) {
        node_ptr c = make_difference(b, a->a);
        return visit(c);
      }

      // a + (-b) -> a - b
      if (b->is_negate()) {
        node_ptr c = make_difference(a, b->a);
        return visit(c);
      }

      node->a = a;
      node->b = b;
      return node;
    }

    constexpr auto operator()(tags::difference, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a);
      node_ptr b = visit(node->b);

      // (-a) - b -> - (a + b)
      if (a->is_negate()) {
        node_ptr c = make_sum(a->a, b);
        node_ptr d = make_negate(c);
        return visit(d);
      }

      // a - (-b) -> a + b
      if (b->is_negate()) {
        node_ptr c = make_sum(a, b->a);
        return visit(c);
      }

      node->a = a;
      node->b = b;
      return node;
    }

    constexpr auto operator()(tags::negate, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a);

      if (a->is_negate()) {
        return a->a;
      }

      node->a = a;
      return node;
    }
  };
}
