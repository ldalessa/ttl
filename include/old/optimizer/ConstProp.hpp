#pragma once

#include "ttl/Tag.hpp"
#include "ttl/optimizer/Nodes.hpp"
#include "ttl/optimizer/Transform.hpp"

#if defined(__GNUC__) && !defined(__clang__)
#include <cmath>
#endif

namespace ttl::optimizer
{
  struct ConstProp : Transform<ConstProp>
  {
    using Transform<ConstProp>::operator();
    using Transform<ConstProp>::visit;

    constexpr auto operator()(tags::sum, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a);
      node_ptr b = visit(node->b);

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
      node_ptr a = visit(node->a);
      node_ptr b = visit(node->b);

      if (b->is_zero()) {
        return a;
      }

      if (a->is_zero()) {
        return make_negate(b);
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
      node_ptr a = visit(node->a);
      node_ptr b = visit(node->b);

      // ? * b -> b * ?
      if (b->is_immediate()) {
        swap(a, b);
      }

      // do some simple optimization
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

      // a * b = [ab]
      if (a->is_immediate() and b->is_immediate()) {
        return a * b;
      }

      // Makes sure the left child has a constant `a`, if there is one.
      // ? * (b · ?) -> (b · ?) * ?
      if (b->is_multiplication() and b->a->is_immediate()) {
        swap(a, b);
      }

      // If neither a nor b is a multiplication then we're done.
      if (!a->is_multiplication() and !b->is_multiplication()) {
        node->a = a;
        node->b = b;
        return node;
      }

      // If a is a multiplication but doesn't have a constant we're done.
      if (a->is_multiplication() and !a->a->is_immediate()) {
        node->a = a;
        node->b = b;
        return node;
      }

      // At this point, if a isn't a constant then we're done.
      if (!a->is_multiplication() and !a->is_immediate()) {
        assert(!b->is_immediate());
        assert(!b->is_multiplication() or !b->a->is_immediate());
        node->a = a;
        node->b = b;
        return node;
      }

      // Handle trees of the form: (a · x) * y
      if (!b->is_multiplication()) {
        if (b->is_immediate()) {
          // (a · x) * b -> [ab] · x
          a->a = a->a * b;
          return visit(a);
        }

        // (a · x) * y -> a * (y · x)  *maybe transposes x and y*
        node->a = a->a;
        a->a = b;
        node->b = a;
        return visit(node);
      }

      // Handle trees of the form: a * (x · y)
      if (!a->is_multiplication()) {
        // a * (b · y) -> [ab] · y
        if (b->a->is_immediate()) {
          b->a = a * b->a;
          return visit(b);
        }

        // a * (x · y)
        node->a = a;
        node->b = b;
        return node;
      }

      // Handle trees of the form: (a · y) * (z · w)
      if (!b->a->is_immediate())
      {
        // (a · y) * (z · w) -> a * ((z · w) · y)
        node->a = a->a;
        a->a = b;
        node->b = visit(a);
        return visit(node);
      }

      // Handle trees of the form: (a · y) * (b · w)
      if (b->a->is_immediate())
      {
        // (a / y) * (b / w) -> [ab] / (y * w)
        if (a->is_ratio() and b->is_ratio())
        {
          node->a = a->b;
          node->b = b->b;
          a->a = a->a * b->a;
          a->b = node;
          return visit(a);
        }

        // (a / y) * (b * w) -> [ab] * (w / y)
        if (a->is_ratio()) {
          node->a = a->a * b->a;
          a->a = b->b;
          node->b = a;
          return visit(node);
        }

        // (a * y) * (b · w) -> [ab] * (y · w)
        // handles both ·
        node->a = a->a * b->a;
        b->a = a->b;
        node->b = b;
        return visit(node);
      }

      assert(false);
      __builtin_unreachable();
    }

    constexpr auto operator()(tags::ratio, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a);
      node_ptr b = visit(node->b);

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
        node_ptr c = make_product(inverse(b), a);
        visit(c);
      }

      if (a->is_immediate() and !b->is_multiplication()) {
        node->a = a;
        node->b = b;
        return node;
      }

      if (a->is_immediate() and !b->a->is_immediate()) {
        node->a = a;
        node->b = b;
        return node;
      }

      if (!a->is_multiplication() and !b->is_multiplication()) {
        node->a = a;
        node->b = b;
        return node;
      }

      // Handle (x · y) / z
      if (a->is_multiplication() and !b->is_multiplication()) {
        // (a · y) / z -> a · (x / y)
        if (a->a->is_immediate()) {
          node->a = a->b;                       // (x / y)
          a->b = node;                          // a · (x / y)
          return visit(a);
        }

        node->a = a;
        node->b = b;
        return node;
      }

      // Handle x / (b · y)
      if (!a->is_multiplication() and b->is_multiplication())
      {
        // a / (b · y) -> [a/b] · y
        if (a->is_immediate() and b->a->is_immediate()) {
          b->a = a / b->a;
          return visit(b);
        }

        if (b->a->is_immediate())
        {
          // x / (a / y) -> 1/a * (x * y)
          if (b->is_ratio())
          {
            node_ptr c = make_product(a, b->b); // (x * y)
            node_ptr d = make_product(inverse(b->a), c);
            return visit(d);
          }

          // x / (a * y) -> 1/a * (x/y)
          node->b = b->b;                     // (x / y)
          b->a = inverse(b->a);               // 1/a *
          b->b = node;                        // 1/a * (x / y)
          return visit(b);
        }

        node->a = a;
        node->b = b;
        return node;
      }

      // (a · y) / (z · w)
      if (a->a->is_immediate() and !b->a->is_immediate())
      {
        // (a / y) / (z / w) -> a * (w / (y * z))
        if (a->is_ratio() and b->is_ratio())
        {
          swap(b->a, b->b);
          b->b = make_product(a->b, b->b);
          node_ptr c = make_product(a->a, b);
          return visit(c);
        }

        // (a / y) / (z * w) -> a / (y * (z * w))
        if (a->is_ratio())
        {
          node->a = a->a;
          node->b = make_product(a->b, b);
          return visit(node);
        }

        // (a * y) / (z / w) -> a * ((y * w) / z)
        if (b->is_ratio())
        {
          swap(b->a, b->b);
          b->a = make_product(a->b, b->a);
          a->b = b;
          return visit(a);
        }

        // (a * y) / (z * w) -> a * (y / (z * w))
        node->a = a->b;
        node->b = b;
        a->b = node;
        return visit(a);
      }

      // (a · y) / (b · w)
      assert(b->a->is_immediate());

      // (a / y) / (b / w) -> [a/b] / (y * w)
      if (a->is_ratio() and b->is_ratio()) {
        node->a = a->a / b->a;
        node->b = make_product(a->b, b->b);
        return visit(node);
      }

      // (a / y) / (b * w) -> [ab] * (w / y)
      if (a->is_ratio()) {
        b->a = a->a * b->a;
        a->a = b->a;
        b->b = a;
        return visit(b);
      }

      // (a * y) / (b / w) -> [ab] * (y / w)
      if (b->is_ratio()) {
        a->a = a->a * b->a;
        b->a = a->b;
        a->b = b;
        return visit(a);
      }

      // (a * y) / (b * w) -> [ab] / (y * w)
      node->a = a->a * a->b;
      b->a = a->b;
      node->b = b;
      return visit(node);
    }

    constexpr auto operator()(tags::pow, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a);
      node_ptr b = visit(node->b);

      if (b->is_zero()) {
        return make_rational(1);
      }

      if (b->is_one()) {
        return a;
      }

      if (a->is_zero()) {
        return a;
      }

      if (a->tag == RATIONAL and b->tag == RATIONAL) {
        return make_rational(pow(a->q, b->q));
      }

      // gcc can do doubles
#if defined(__GNUC__) && !defined(__clang__)
      if (a->is_immediate() and b->is_immediate()) {
        return make_double(std::pow(a->as_double(), b->as_double()));
      }
#endif

      node->a = a;
      node->b = b;
      return node;
    }

    constexpr auto operator()(tags::negate, node_ptr const& node) const
      -> node_ptr
    {
      node_ptr a = visit(node->a);

      if (a->is_immediate()) {
        return -a;
      }

      if (a->is_multiplication() and a->a->is_immediate()) {
        return make_product(-(a->a), a->b);
      }

      node->a = a;
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
  };
}
