#pragma once

#include "ttl/Tag.hpp"
#include "ttl/optimizer/Nodes.hpp"

namespace ttl::optimizer
{
  template <class Derived>
  struct Transform
  {
    constexpr Derived const& self() const
    {
      return *static_cast<Derived const*>(this);
    }

    constexpr Derived& self()
    {
      return *static_cast<Derived*>(this);
    }

    constexpr static auto make_sum(node_ptr const& a, node_ptr const& b)
      -> node_ptr
    {
      return new Node(tag_v<SUM>, a, b);
    }

    constexpr static auto make_difference(node_ptr const& a, node_ptr const& b)
      -> node_ptr
    {
      return new Node(tag_v<DIFFERENCE>, a, b);
    }

    constexpr static auto make_product(node_ptr const& a, node_ptr const& b)
      -> node_ptr
    {
      return new Node(tag_v<PRODUCT>, a, b);
    }

    constexpr static auto make_ratio(node_ptr const& a, node_ptr const& b)
      -> node_ptr
    {
      return new Node(tag_v<RATIO>, a, b);
    }

    constexpr static auto make_pow(node_ptr const& a, Rational q)
      -> node_ptr
    {
      return new Node(tag_v<POW>, a, make_rational(q));
    }

    constexpr static auto make_negate(node_ptr const& a)
      -> node_ptr
    {
      return new Node(tag_v<NEGATE>, a);
    }

    constexpr static auto make_partial(node_ptr const& a, Index index)
    {
      return new Node(tag_v<PARTIAL>, a, index);
    }

    constexpr static auto make_rational(Rational q)
      -> node_ptr
    {
      return new Node(tag_v<RATIONAL>, q);
    }

    constexpr static auto make_double(double d)
      -> node_ptr
    {
      return new Node(tag_v<DOUBLE>, d);
    }

    /// Tag-dispatch for the node type.
    template <class... Args>
    constexpr auto visit(node_ptr const& node, Args&&... args) const
      -> node_ptr
    {
      return ttl::visit(node->tag, self(), node, std::forward<Args>(args)...);
    }

    /// Outer entry point.
    template <class... Args>
    constexpr auto operator()(node_ptr const& node, Args&&... args) const
      -> node_ptr
    {
      node_ptr root = visit(node, std::forward<Args>(args)...);
      root->update_tree_constants();
      return root;
    }

    /// Generic binary visit.
    template <class... Args>
    constexpr auto operator()(tags::binary, node_ptr const& node, Args&&... args) const
      -> node_ptr
    {
      node->a = visit(node->a, std::forward<Args>(args)...);
      node->b = visit(node->b, std::forward<Args>(args)...);
      return node;
    }

    /// Generic unary visit.
    template <class... Args>
    constexpr auto operator()(tags::unary, node_ptr const& node, Args&&... args) const
      -> node_ptr
    {
      node->a = visit(node->a, std::forward<Args>(args)...);
      return node;
    }

    /// Generic leaf visit.
    template <class... Args>
    constexpr auto operator()(tags::leaf, node_ptr const& node, Args&&...) const
      -> node_ptr
    {
      return node;
    }
  };
}
