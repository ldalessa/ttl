#pragma once

#include "ttl/Tag.hpp"
#include "ttl/optimizer/Nodes.hpp"
#include "ttl/optimizer/Transform.hpp"

namespace ttl::optimizer
{
  struct LowerPartials : Transform<LowerPartials>
  {
    using Transform<LowerPartials>::operator();
    using Transform<LowerPartials>::visit;

    /// We won't have an outer index at the outermost partial.
    constexpr auto operator()(tags::partial, node_ptr const& node) const
      -> node_ptr
    {
      if (node->a->is_constant()) {
        return make_rational(0);
      }
      else {
        return visit(node->a, node->tensor_index);
      }
    }

    constexpr auto operator()(tags::partial, node_ptr const& node, Index dx) const
      -> node_ptr
    {
      if (node->a->is_constant()) {
        return make_rational(0);
      }
      else {
        return visit(node->a, node->tensor_index + dx);
      }
    }

    constexpr auto operator()(tags::tensor, node_ptr const& node, Index dx) const
      -> node_ptr
    {
      if (node->is_constant()) {
        return make_rational(0);
      }

      node_ptr a = node->copy_tree();
      a->tensor_index += dx;
      return a;
    }

    constexpr auto operator()(tags::sum, node_ptr const& node, Index dx) const
      -> node_ptr
    {
      // ∂(a + v) = ∂(v)
      if (node->a->is_constant()) {
        return visit(node->b, dx);
      }

      // ∂(u + a) = ∂(u)
      if (node->b->is_constant()) {
        return visit(node->a, dx);
      }

      node->a = visit(node->a, dx);
      node->b = visit(node->b, dx);
      return node;
    }

    constexpr auto operator()(tags::difference, node_ptr const& node, Index dx) const
      -> node_ptr
    {
      // ∂(a - v) = -∂(v)
      if (node->a->is_constant()) {
        return make_negate(visit(node->b, dx));
      }

      // ∂(u - a) = ∂(u)
      if (node->b->is_constant()) {
        return visit(node->a, dx);
      }

      node->a = visit(node->a, dx);
      node->b = visit(node->b, dx);
      return node;
    }

    constexpr auto operator()(tags::product, node_ptr const& node, Index dx) const
      -> node_ptr
    {
      // ∂(av) = a∂(v)
      if (node->a->is_constant()) {
        node->b = visit(node->b, dx);
        return node;
      }

      // ∂(ub) = ∂(u)b
      if (node->b->is_constant()) {
        node->a = visit(node->a, dx);
        return node;
      }

      // ∂(uv) = ∂(u)v + u∂(v)
      node_ptr  u = visit(node->a);
      node_ptr  v = visit(node->b);
      node_ptr du = visit(node->a->copy_tree(), dx);
      node_ptr dv = visit(node->b->copy_tree(), dx);

      node_ptr duv = make_product(du, v);
      node_ptr udv = make_product(u, dv);
      return make_sum(duv, udv);
    }

    constexpr auto operator()(tags::ratio, node_ptr const& node, Index dx) const
      -> node_ptr
    {
      // ∂(a/v) = -a∂(v)/(v^2)
      if (node->a->is_constant()) {
        node_ptr  v = visit(node->b);
        node_ptr dv = visit(node->b->copy_tree(), dx);
        node_ptr v2 = make_pow(v, Rational(2));
        node->a = make_product(node->a, dv);
        node->b = v2;
        return make_negate(node);
      }

      // ∂(u/b) = ∂(u)/b
      if (node->b->is_constant()) {
        node->a = visit(node->a, dx);
        return node;
      }

      // ∂(u/v) = (∂(u)v - u∂(v))/(v^2)
      node_ptr  u = visit(node->a);
      node_ptr  v = visit(node->b);
      node_ptr du = visit(node->a->copy_tree(), dx);
      node_ptr dv = visit(node->b->copy_tree(), dx);
      node_ptr v2 = make_pow(v->copy_tree(), Rational(2));

      node_ptr duv = make_product(du, v);
      node_ptr udv = make_product(u, dv);
      node->a = make_difference(duv, udv);
      node->b = v2;
      return node;
    }

    constexpr auto operator()(tags::pow, node_ptr const& node, Index dx) const
      -> node_ptr
    {
      // ∂(u^n) = n * (∂(u))^n - 1
      assert(node->b->tag == RATIONAL);
      node_ptr  n = node->b->copy_tree();;
      node->b->q -= Rational(1);
      node_ptr du = visit(node->a, dx);
      node->a = du;
      return make_product(n, node);
    }

    constexpr auto operator()(tags::exp, node_ptr const& node, Index dx) const
      -> node_ptr
    {
      node_ptr a = visit(node->a);
      node_ptr b = visit(node->a->copy_tree(), dx);
      node->a = a;
      return make_product(node, b);
    }
  };
}
