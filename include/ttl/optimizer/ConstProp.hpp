#pragma once

#include "ttl/optimizer/Nodes.hpp"

namespace ttl::optimizer
{
  struct ConstProp
  {
    constexpr auto operator()(node_ptr node) const -> node_ptr
    {
      return node_ptr(node.visit(*this));
    }

    constexpr auto operator()(Sum* node) const -> node_ptr
    {
      node_ptr a = node->a.visit(*this);
      node_ptr b = node->b.visit(*this);
      node->a = a;
      node->b = b;

      if (a->is_zero()) {
        return b;
      }

      if (b->is_zero()) {
        return a;
      }

      bool alit = tag_is_literal(a->tag);
      bool blit = tag_is_literal(b->tag);

      if (alit and blit) {
        return combine(SUM, *a, *b);
      }

      return node_ptr(node);
    }

    constexpr auto operator()(Difference* node) const -> node_ptr
    {
      node_ptr a = node->a.visit(*this);
      node_ptr b = node->b.visit(*this);
      node->a = a;
      node->b = b;

      if (b->is_zero()) {
        return a;
      }

      if (a->is_zero()) {
        return node_ptr(new Negate(b));
      }

      bool alit = tag_is_literal(a->tag);
      bool blit = tag_is_literal(b->tag);

      if (alit and  blit) {
        return combine(DIFFERENCE, *a, *b);
      }

      return node_ptr(node);
    }

    constexpr auto operator()(Product* node) const -> node_ptr
    {
      node_ptr a = node->a.visit(*this);
      node_ptr b = node->b.visit(*this);

      if (a->is_zero()) {
        return a;
      }

      if (b->is_zero()) {
        return b;
      }

      if (a->is_one()) {
        return b;
      }

      if (b->is_one()) {
        return b;
      }

      // bool alit = a->tag.is_literal();
      // bool blit = b->tag.is_literal();

      // // a * b = ab
      // if (alit and blit) {
      //   return combine(PRODUCT, *a, *b);
      // }

      // // a * (b · x) -> ab · x
      // if (alit and b-tag.is_multiplication()) {
      //   if (b->a->tag.is_literal()) {
      //     b->a = combine(PRODUCT, a, b->a);
      //     return b;
      //   }
      // }

      // // (a · x) * b -> ab · x
      // if (blit and a->tag.is_multiplication()) {
      //   if (a->a->tag.is_literal()) {
      //     a->a = combine(PRODUCE, a->a, b);
      //     return a;
      //   }
      // }

      // // (x * b) -> (b * a)  [canonical]
      // if (blit) {
      //   node->a = b;
      //   node->b = a;
      //   return node_ptr(node);
      // }

      // if (!a->tag.is_multiplication() or !b->tag.is_multiplication()) {
        node->a = a;
        node->b = b;
        return node_ptr(node);
      // }
    }

    constexpr auto operator()(Ratio* node) const -> node_ptr
    {
      node_ptr a = node->a.visit(*this);
      node_ptr b = node->b.visit(*this);

      if (a->is_zero()) {
        return a;
      }

      if (b->is_zero()) {
        ttl_assert(false, "divide by zero");
      }

      if (b->is_one()) {
        return a;
      }

      node->a = a;
      node->b = b;
      return node_ptr(node);
    }

    constexpr auto operator()(Pow* node) const -> node_ptr
    {
      node_ptr a = node->a.visit(*this);
      node_ptr b = node->b.visit(*this);

      if (b->is_zero()) {
        return node_ptr::one();
      }

      if (b->is_one()) {
        return a;
      }

      if (a->is_zero()) {
        return a;
      }

      bool alit = tag_is_literal(a->tag);
      bool blit = tag_is_literal(b->tag);
      assert(blit);

      if (alit) {
        return combine(POW, *a, *b);
      }

      node->a = a;
      node->b = b;
      return node_ptr(node);
    }

    constexpr auto operator()(Unary* node) const -> node_ptr
    {
      node->a = node->a.visit(*this);
      return node_ptr(node);
    }

    constexpr auto operator()(Bind* bind) const -> node_ptr
    {
      if (std::is_constant_evaluated()) {
        assert(false);
      }
      return {};
    }

    constexpr auto operator()(Leaf* leaf) const -> node_ptr
    {
      return node_ptr(leaf);
    }
  };
}
