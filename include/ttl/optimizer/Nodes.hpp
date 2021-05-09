#pragma once

#include "ttl/Tag.hpp"
#include "ttl/Tensor.hpp"

namespace ttl::optimizer
{
  struct Node;

  struct node_ptr
  {
    Node *ptr_ = nullptr;

    constexpr node_ptr() = default;

    /// Initialize from a pointer.
    constexpr node_ptr(Node *ptr)
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
      // increment first prevents early delete
      b.inc();
      dec();
      ptr_ = b.ptr_;
      return *this;
    }

    constexpr node_ptr& operator=(node_ptr&& b)
    {
      if (ptr_ != b.ptr_) {
        dec();
        ptr_ = std::exchange(b.ptr_, nullptr);
      }
      return *this;
    }

    constexpr node_ptr& operator=(Node* ptr);

    constexpr explicit operator bool() const
    {
      return ptr_ != nullptr;
    }

    constexpr friend void swap(node_ptr& a, node_ptr& b) {
      std::swap(a.ptr_, b.ptr_);
    }

    constexpr friend bool operator==(node_ptr const&, node_ptr const&) = default;

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
    node_ptr               a = {};
    node_ptr               b = {};
    Rational               q = { 1 };
    double                 d = 1.0;
    TensorBase const* tensor = nullptr;
    Index       tensor_index = {};
    ScalarIndex scalar_index = {};
    bool            constant = false;

    constexpr Node() = default;

    constexpr Node(Tag tag,
                   node_ptr $a,
                   node_ptr $b,
                   Rational const& q,
                   double d,
                   TensorBase const* tensor,
                   Index const& tensor_index,
                   ScalarIndex const& scalar_index,
                   bool constant)
        : tag(tag)
        , a($a) // would like to move here but constexpr bugs in gcc-10.3,clang-11
        , b($b) // would like to move here but constexpr bugs in gcc-10.3,clang-11
        , q(q)
        , d(d)
        , tensor(tensor)
        , tensor_index(tensor_index)
        , scalar_index(scalar_index)
        , constant(constant)
    {
      assert(ispow2(tag));
      if (tag_is_binary(tag)) {
        assert(a and b);
      }
      if (tag_is_unary(tag)) {
        assert(a and !b);
      }
      if (tag_is_leaf(tag)) {
        assert(!a and !b);
      }
    }

    constexpr Node(tags::is_binary auto tag, node_ptr const& a, node_ptr const& b)
        : tag(tag.id)
        , a(a)
        , b(b)
        , constant(a->constant && b->constant)
    {
      assert(ispow2(this->tag));
    }

    constexpr Node(tags::partial, node_ptr const& a, Index i)
        : tag(PARTIAL)
        , a(a)
        , tensor_index(i)
    {
    }

    constexpr Node(tags::rational, Rational q)
        : tag(RATIONAL)
        , q(q)
        , constant(true)
    {
    }

    constexpr Node(tags::floating_point, double d)
        : tag(DOUBLE)
        , d(d)
        , constant(true)
    {
    }

    constexpr Node(tags::negate, node_ptr a)
        : tag(NEGATE)
        , a(a)
        , constant(a->constant)
    {
    }

    constexpr node_ptr copy_tree() const
    {
      Node *copy = new Node();
      copy->tag = this->tag;
      copy->q = this->q;
      copy->d = this->d;
      copy->tensor = this->tensor;
      copy->tensor_index = this->tensor_index;
      copy->scalar_index = this->scalar_index;
      copy->constant = this->constant;
      if (a) copy->a = this->a->copy_tree();
      if (b) copy->b = this->b->copy_tree();
      assert(ispow2(copy->tag));
      return node_ptr(copy);
    }

    constexpr bool update_tree_constants()
    {
      if (a and b) {
        assert(is_binary());
        bool aa = a->update_tree_constants();
        bool bb = b->update_tree_constants();
        constant = aa and bb;
        return constant;
      }

      if (a) {
        assert(is_unary());
        constant = a->update_tree_constants();
        return constant;
      }

      assert(is_leaf());
      return constant;
    }

    constexpr bool verify_tree_constants() const
    {
      if (a and b) {
        assert(is_binary());
        bool aa = a->verify_tree_constants();
        bool bb = b->verify_tree_constants();
        assert(constant == (aa and bb));
        return constant;
      }

      if (a) {
        assert(is_unary());
        bool aa = a->verify_tree_constants();
        assert(constant == aa);
        return constant;
      }

      assert(is_leaf());
      return constant;
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

    constexpr bool is_multiplication() const
    {
      return tag_is_multiplication(tag);
    }

    constexpr bool is_ratio() const
    {
      return tag == RATIO;
    }

    constexpr bool is_negate() const
    {
      return tag == NEGATE;
    }

    constexpr auto outer() const -> Index
    {
      if (is_binary()) {
        return tag_outer(tag, a->outer(), b->outer());
      }
      if (is_unary()) {
        return tag_outer(tag, a->outer(), tensor_index);
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

    constexpr double as_double() const
    {
      assert(is_immediate());
      return d * as<double>(q);
    }
  };

  constexpr node_ptr& node_ptr::operator=(Node* ptr)
  {
    if (ptr) {
      ++ptr->count;
    }
    dec();
    ptr_ = ptr;
    return *this;
  }


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

  constexpr node_ptr operator-(node_ptr const& a)
  {
    assert(a->is_immediate());
    node_ptr c = new Node();
    if (a->d != 1.0) {
      c->d = -a->d;
      c->tag = DOUBLE;
    }
    else {
      c->q = -a->q;
      c->tag = RATIONAL;
    }
    c->constant = true;
    return c;
  }

  constexpr node_ptr operator+(node_ptr const& a, node_ptr const& b)
  {
    assert(a->is_immediate() and b->is_immediate());
    node_ptr c = new Node();
    c->d = a->as_double() + b->as_double();
    c->tag = DOUBLE;
    c->constant = true;
    return c;
  }

  constexpr node_ptr operator-(node_ptr const& a, node_ptr const& b)
  {
    assert(a->is_immediate() and b->is_immediate());
    node_ptr c = new Node();
    c->d = a->as_double() - b->as_double();
    c->tag = DOUBLE;
    c->constant = true;
    return c;
  }

  constexpr node_ptr operator*(node_ptr const& a, node_ptr const& b)
  {
    assert(a->is_immediate() and b->is_immediate());
    node_ptr c = new Node();
    c->d = a->d * b->d;
    c->q = a->q * b->q;
    c->tag = (c->d != 1.0) ? DOUBLE : RATIONAL;
    c->constant = true;
    return c;
  }

  constexpr node_ptr operator/(node_ptr const& a, node_ptr const& b)
  {
    assert(a->is_immediate() and b->is_immediate());
    node_ptr c = new Node();
    c->d = a->d / b->d;
    c->q = a->q / b->q;
    c->tag = (c->d != 1.0) ? DOUBLE : RATIONAL;
    c->constant = true;
    return c;
  }

  constexpr node_ptr inverse(node_ptr const& a)
  {
    assert(a->is_immediate());
    node_ptr c = new Node();
    c->d = 1.0 / a->d;
    c->q = a->q.inverse();
    c->tag = (c->d != 1.0) ? DOUBLE : RATIONAL;
    c->constant = true;
    return c;
  }

  constexpr auto visit(node_ptr const& node, auto&& op, auto&&... args)
    -> decltype(auto)
  {
    return visit(node->tag, std::forward<decltype(op)>(op), node, std::forward<decltype(args)>(args)...);
  }
}
