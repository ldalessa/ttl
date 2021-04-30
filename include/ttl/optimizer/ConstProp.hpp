#pragma once

#include "ttl/Tag.hpp"
#include "ttl/optimizer/Nodes.hpp"

namespace ttl::optimizer
{
  struct ConstProp
  {
    constexpr auto operator()(node_ptr const& node) const
      -> node_ptr
    {
      return visit(node, *this);
    }

    constexpr auto operator()(tags::sum, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a, *this);
      node_ptr b = visit(node->b, *this);

      if (a->is_zero()) {
        return b;
      }

      if (b->is_zero()) {
        return a;
      }

      if (a->is_immediate() and b->is_immediate()) {
        return a + b;
      }

      node->a = std::move(a);
      node->b = std::move(b);
      return node;
    }

    constexpr auto operator()(tags::difference, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a, *this);
      node_ptr b = visit(node->b, *this);

      if (b->is_zero()) {
        return a;
      }

      if (a->is_zero()) {
        return node_ptr(new Node(tag_v<NEGATE>, std::move(b)));
      }

      if (a->is_immediate() and b->is_immediate()) {
        return a - b;
      }

      node->a = std::move(a);
      node->b = std::move(b);
      return node;
    }

    constexpr auto operator()(tags::product tag, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a, *this);
      node_ptr b = visit(node->b, *this);

      if (a->is_one()) {
        return b;
      }

      if (b->is_one()) {
        return a;
      }

      if (a->is_zero()) {
        return a;
      }

      if (b->is_zero()) {
        return b;
      }

      // a * b = ab
      if (a->is_immediate() and b->is_immediate()) {
        return a * b;
      }

      // a * (b · x) -> ab · x
      if (a->is_immediate() and b->is_multiplication()) {
        if (b->a->is_immediate()) {
          b->a = a * b->a;
          return visit(b, *this);
        }
      }

      // (a · x) * b -> ab · x
      if (a->is_multiplication() and b->is_immediate()) {
        if (a->a->is_immediate()) {
          a->a = a->a * b;
          return visit(a, *this);
        }
      }

      // (x * b) -> (b * x)  [canonical]
      if (b->is_immediate()) {
        node->a = b;
        node->b = a;
        (*this)(tag, node);
      }

      if (!a->is_multiplication() or !b->is_multiplication()) {
        node->a = a;
        node->b = b;
        return node;
      }

      // (a · y) * (b · w)
      if (a->a->is_immediate() and b->a->is_immediate())
      {
        // (a / y) * (b / w) -> ab / yw
        if (a->is_ratio() and b->is_ratio()) {
          node->a = a->b;                       // y
          node->b = b->b;                       // yw
          a->a = a->a * b->a;                   // ab
          a->b = node;                          // ab / yw
          visit(a, *this);
        }

        // (a / y) * (b * w) -> ab * (w / y)
        if (a->is_ratio()) {
          node->a = a->a * b->a;                // ab
          node->b = a;                          // ab * (_ / y)
          a->a = b->b;                          // ab * (w / y)
          return visit(a, *this);
        }

        // (a * y) * (b · w) -> ab * (y · w)
        node->a = a->a * b->a;                  // ab
        node->b = b;                            // ab * (_ · w)
        b->a = a->b;                            // ab * (y · w)
        return visit(b, *this);
      }

      // (a · y) * (z · w)
      if (a->a->is_immediate())
      {
        // (a / y) * (z / w) -> a * (z / (y * w))
        if (a->is_ratio() and b->is_ratio()) {
          node->a = a->a;                              // a
          b->b = new Node(tag_v<PRODUCT>, a->b, b->b); // z / (y * w)
          node->b = a;                                 // a * (z / (y * w))
          return visit(node, *this);
        }

        // (a / y) * (z * w) -> a * ((z * w) / y)
        if (a->is_ratio()) {
          node->a = a->a;                       // a
          a->a = b;                             // (z * w) / y
          node->b = a;                          // a * ((z * w) / y)
          return visit(node, *this);
        }

        // (a * y) * (z · w) -> a * ((y * z) · w)
        node->a = a->a;                         // a
        a->a = b->a;                            // (z * y)
        b->a = a;                               // (z * y) · w
        node->b = b;                            // a * ((z * y) · w)
        return visit(node, *this);
      }

      // (x · y) * (b · w)
      if (b->a->is_immediate())
      {
        // (x / y) * (b / w) -> b * (x / (y * w))
        if (a->is_ratio() and b->is_ratio()) {
          node->a = b->a;                              // b
          a->b = new Node(tag_v<PRODUCT>, a->b, b->b); // x / (y * w)
          node->b = a;                                 // b * (z / (y * w))
          return visit(node, *this);
        }

        // (x / y) * (b * w) -> b * ((x * w) / y)
        if (a->is_ratio()) {
          node->a = b->a;                       // b
          b->a = a->a;                          // (x * w)
          a->a = b;                             // (x * w) / y
          node->b = a;                          // a * ((x * w) / y)
          return visit(node, *this);
        }

        // (x * y) * (b · w) -> b * ((x * y) · w)
        node->a = b->a;                         // b
        b->a = a;                               // b * ((x * y) · w)
        visit(node, *this);
      }

      node->a = a;
      node->b = b;
      return node;
    }

    constexpr auto operator()(tags::ratio, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a, *this);
      node_ptr b = visit(node->b, *this);

      if (a->is_zero()) {
        return a;
      }

      if (b->is_zero()) {
        ttl_assert(false, "divide by zero");
      }

      if (b->is_one()) {
        return a;
      }

      if (a->is_immediate() and b->is_immediate()) {
        return a / b;
      }

      if (b->is_immediate()) {
        node_ptr c = new Node(tag_v<PRODUCT>, inverse(b), a);
        visit(c, *this);
      }

      if (a->is_multiplication()) {
        // (a · x) / y -> a · (x / y)
        if (a->a->is_immediate()) {
          node->a = a->b;                       // (x / y)
          a->b = node;                          // a · (x / y)
          visit(a, *this);
        }
      }

      if (b->is_multiplication()) {
        // x / (a · y)
        if (b->a->is_immediate()) {
          // x / (a / y) -> 1/a * (x * y)
          if (b->is_ratio()) {
            node_ptr c = new Node(tag_v<PRODUCT>, node->a, b->b); // (x * y)
            node_ptr d = new Node(tag_v<PRODUCT>, inverse(b->a), c);
            visit(d, *this);
          }

          // x / (a * y) -> 1/a * (x/y)
          node->b = b->b;                     // (x / y)
          node_ptr c = new Node(tag_v<PRODUCT>, inverse(b->a), node);
          visit(c, *this);
        }
      }

      node->a = std::move(a);
      node->b = std::move(b);
      return node;
    }

    constexpr auto operator()(tags::pow, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a, *this);
      node_ptr b = visit(node->b, *this);

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

      node->a = std::move(a);
      node->b = std::move(b);
      return node;
    }

    constexpr auto operator()(tags::unary, node_ptr const& node) const
      -> node_ptr
    {
      node->a = visit(node->a, *this);
      return node;
    }

    constexpr auto operator()(tags::bind, node_ptr const& bind) const
      -> node_ptr
    {
      if (std::is_constant_evaluated()) {
        assert(false);
      }
      return node_ptr();
    }

    constexpr auto operator()(tags::leaf, node_ptr const& leaf) const
      -> node_ptr
    {
      return leaf;
    }
  };
}
