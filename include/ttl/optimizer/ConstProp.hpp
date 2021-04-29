#pragma once

#include "ttl/Tag.hpp"
#include "ttl/optimizer/Nodes.hpp"

namespace ttl::optimizer
{
  struct ConstProp
  {
    constexpr auto operator()(node_ptr node) const
      -> node_ptr
    {
      return visit(node, *this);
    }

    constexpr auto operator()(tags::sum, node_ptr node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a(), *this);
      node_ptr b = visit(node->b(), *this);

      if (a->is_zero()) {
        return b;
      }

      if (b->is_zero()) {
        return a;
      }

      // bool alit = a->is_literal();
      // bool blit = b->is_literal();

      // if (alit and blit) {
      //   return node_ptr{*a * *b};
      // }

      node->a(std::move(a));
      node->b(std::move(b));
      return node;
    }

    constexpr auto operator()(tags::difference, node_ptr node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a(), *this);
      node_ptr b = visit(node->b(), *this);

      if (b->is_zero()) {
        return a;
      }

      if (a->is_zero()) {
        return node_ptr(new Node(tag_v<NEGATE>, std::move(b)));
      }

      // bool alit = tag_is_literal(a->tag);
      // bool blit = tag_is_literal(b->tag);

      // if (alit and  blit) {
      //   return combine(DIFFERENCE, *a, *b);
      // }

      node->a(std::move(a));
      node->b(std::move(b));
      return node;
    }

    constexpr auto operator()(tags::product, node_ptr node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a(), *this);
      node_ptr b = visit(node->b(), *this);

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
      node->a(std::move(a));
      node->b(std::move(b));
      return node;
      // }
    }

    constexpr auto operator()(tags::ratio, node_ptr node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a(), *this);
      node_ptr b = visit(node->b(), *this);

      if (a->is_zero()) {
        return a;
      }

      if (b->is_zero()) {
        ttl_assert(false, "divide by zero");
      }

      if (b->is_one()) {
        return a;
      }

      node->a(std::move(a));
      node->b(std::move(b));
      return node;
    }

    constexpr auto operator()(tags::pow, node_ptr node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a(), *this);
      node_ptr b = visit(node->b(), *this);

      if (b->is_zero()) {
        return node_ptr(new Node(tag_v<RATIONAL>, 1));
      }

      if (b->is_one()) {
        return a;
      }

      if (a->is_zero()) {
        return a;
      }

      // bool alit = tag_is_literal(a->tag);
      // bool blit = tag_is_literal(b->tag);
      // assert(blit);

      // if (alit) {
      //   return combine(POW, *a, *b);
      // }

      node->a(std::move(a));
      node->b(std::move(b));
      return node;
    }

    constexpr auto operator()(tags::unary, node_ptr node) const
      -> node_ptr
    {
      node->a(visit(node->a(), *this));
      return node;
    }

    constexpr auto operator()(tags::bind, node_ptr bind) const
      -> node_ptr
    {
      if (std::is_constant_evaluated()) {
        assert(false);
      }
      return node_ptr();
    }

    constexpr auto operator()(tags::leaf, node_ptr leaf) const
      -> node_ptr
    {
      return leaf;
    }
  };
}
