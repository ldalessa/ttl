#pragma once

#include "ttl/Tag.hpp"
#include <raberu.hpp>

namespace ttl::optimizer
{
  namespace kw
  {
    using namespace rbr::literals;
    inline constexpr auto a = "a"_kw;
    inline constexpr auto b = "b"_kw;
    inline constexpr auto q = "q"_kw;
    inline constexpr auto d = "d"_kw;
    inline constexpr auto tensor = "tensor"_kw;
    inline constexpr auto tensor_index = "tensor_index"_kw;
    inline constexpr auto scalar_index = "scalar_index"_kw;
  }

  struct Node;

  struct node_ptr
  {
    Node *ptr_ = nullptr;

    constexpr node_ptr() = default;

    /// Initialize from a pointer.
    constexpr explicit node_ptr(Node *ptr)
        : ptr_(ptr)
    {
      inc();
    }

    constexpr node_ptr(node_ptr const& b)
        : ptr_(b.ptr_)
    {
      inc();
    }

    constexpr node_ptr(node_ptr&& b)
        : ptr_(std::exchange(b.ptr_, nullptr))
    {
    }

    constexpr ~node_ptr()
    {
      dec();
    }

    constexpr node_ptr& operator=(node_ptr const& b)
    {
      if (ptr_ != b.ptr_) {
        dec();
        ptr_ = b.ptr_;
        inc();
      }
      return *this;
    }

    constexpr node_ptr& operator=(node_ptr&& b)
    {
      dec();
      ptr_ = std::exchange(b.ptr_, nullptr);
      return *this;
    }

    constexpr node_ptr& operator=(Node *ptr)
    {
      if (ptr_ != ptr) {
        dec();
        ptr_ = ptr;
        inc();
      }
      return *this;
    }

    constexpr explicit operator bool() const
    {
      return ptr_ != nullptr;
    }

    constexpr friend bool operator==(node_ptr const&, node_ptr const&) = default;

    constexpr friend bool operator==(node_ptr const& a, Node* b)
    {
      return a.ptr_ == b;
    }

    constexpr friend bool operator==(Node* a, node_ptr const& b)
    {
      return a == b.ptr_;
    }

    constexpr auto operator*() const -> Node&
    {
      return *ptr_;
    }

    constexpr auto operator->() const -> Node*
    {
      return ptr_;
    }

    constexpr void inc() const;
    constexpr void dec();
  };

  struct Node
  {
    Tag                  tag = NO_TAG;
    int                count = 0;
    node_ptr     children[2] = {};
    Rational               q = { 1 };
    double                 d = 1.0;
    Tensor const*     tensor = nullptr;
    Index       tensor_index = {};
    ScalarIndex scalar_index = {};
    bool            constant = false;

    constexpr Node() = default;

    constexpr Node(Tag tag,
                   node_ptr&& a,
                   node_ptr&& b,
                   Rational const& q,
                   double d,
                   Tensor const* tensor,
                   Index const& tensor_index,
                   ScalarIndex const& scalar_index,
                   bool constant)
        : tag(tag)
        , children { std::move(a), std::move(b) }
        , q(q)
        , d(d)
        , tensor(tensor)
        , tensor_index(tensor_index)
        , scalar_index(scalar_index)
        , constant(constant)
    {
    }

    constexpr Node(tags::rational, Rational q)
        : tag(RATIONAL)
        , q(q)
    {
    }

    constexpr Node(tags::negate, node_ptr a)
        : tag(NEGATE)
    {
      children[0] = std::move(a);
    }

    constexpr auto a() const -> node_ptr const&
    {
      return children[0];
    }

    constexpr auto b() const -> node_ptr const&
    {
      return children[1];
    }

    constexpr void a(node_ptr&& a)
    {
      children[0] = std::move(a);
    }

    constexpr void b(node_ptr&& b)
    {
      children[1] = std::move(b);
    }

    constexpr bool is_binary() const
    {
      return tag_is_binary(tag);
    }

    constexpr bool is_unary() const
    {
      return tag_is_unary(tag);
    }

    constexpr bool is_leaf() const
    {
      return tag_is_leaf(tag);
    }

    constexpr bool is_immediate() const
    {
      return tag_is_immediate(tag);
    }

    constexpr bool is_constant() const
    {
      return constant;
    }

    constexpr auto outer() const -> Index
    {
      if (is_binary()) {
        return tag_outer(tag, a()->outer(), b()->outer());
      }
      if (is_unary()) {
        return tag_outer(tag, a()->outer(), tensor_index);
      }
      return tag_outer(tag, tensor_index);
    }

    constexpr bool is_zero() const
    {
      return is_immediate() && (q == 0 || d == 0.0);
    }

    constexpr bool is_one()  const
    {
      return is_immediate() && q == 1 && d == 1;
    }
  };

  constexpr void node_ptr::inc() const
  {
    if (ptr_) {
      ++ptr_->count;
    }
  }

  constexpr void node_ptr::dec()
  {
    if (ptr_) {
      if (--ptr_->count == 0) {
        delete std::exchange(ptr_, nullptr);
      }
    }
  }

  constexpr auto visit(node_ptr const& node, auto&& op, auto&&... args)
    -> decltype(auto)
  {
    return visit(node->tag, std::forward<decltype(op)>(op), node, std::forward<decltype(args)>(args)...);
  }
}
