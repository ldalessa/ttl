#pragma once

#include "ttl/Rational.hpp"
#include "ttl/Tags.hpp"
#include "ttl/TensorIndex.hpp"

namespace ttl
{
  struct Node
  {
    int count = 0;
    TreeTag tag_;
    constexpr Node(TreeTag tag)
        : tag_(tag)
    {
    }

    constexpr auto tag() const -> TreeTag
    {
      return tag_;
    }

    constexpr virtual void destroy() const = 0;

    constexpr virtual auto size() const -> int
    {
      return 1;
    }

    constexpr virtual auto outer_index() const -> TensorIndex
    {
      return {};
    }

    constexpr virtual auto rank() const -> int
    {
      return outer_index().size();
    }
  };

  struct node_ptr
  {
    Node *ptr_;

    constexpr ~node_ptr()
    {
      dec();
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

    constexpr node_ptr(Node *ptr)
        : ptr_(ptr)
    {
      inc();
    }

    // constexpr node_ptr& operator=(node_ptr const& b)
    // {
    //   // Increment first prevents early delete, code works for self-loops.
    //   b.inc();
    //   dec();
    //   ptr_ = b.ptr_;
    //   return *this;
    // }

    // constexpr node_ptr& operator=(node_ptr&& b)
    // {
    //   if (ptr_ != b.ptr_) {
    //     dec();
    //     ptr_ = std::exchange(b.ptr_, nullptr);
    //   }
    //   return *this;
    // }

    // constexpr node_ptr& operator=(Node* ptr)
    // {
    //   if (ptr) {
    //     ++ptr->count;
    //   }
    //   dec();
    //   ptr_ = ptr;
    //   return *this;
    // }

    constexpr friend auto operator<=>(node_ptr const&, node_ptr const&) = default;

    constexpr auto operator->() const -> Node*
    {
      return ptr_;
    }

    constexpr void inc() const
    {
      if (ptr_) {
        ++ptr_->count;
      }
    }

    constexpr void dec()
    {
      if (ptr_) {
        assert(ptr_->count > 0);
        if (--ptr_->count == 0) {
          std::exchange(ptr_, nullptr)->destroy();
        }
      }
    }
  };

  struct Binary : Node
  {
    node_ptr a;
    node_ptr b;

    constexpr Binary(TreeTag tag, node_ptr a, node_ptr b)
        : Node(tag)
        , a(std::move(a))
        , b(std::move(b))
    {
    }

    constexpr auto size() const -> int override
    {
      return a->size() + b->size() + 1;
    }
  };

  struct Addition : Binary
  {
    constexpr Addition(TreeTag tag, node_ptr a, node_ptr b)
        : Binary(tag, std::move(a), std::move(b))
    {
      assert(permutation(this->a->outer_index(), this->b->outer_index()));
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return a->outer_index();
    }
  };

  struct Sum final : Addition
  {
    constexpr Sum(node_ptr a, node_ptr b)
        : Addition(SUM, std::move(a), std::move(b))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Difference final : Addition
  {
    constexpr Difference(node_ptr a, node_ptr b)
        : Addition(DIFFERENCE, std::move(a), std::move(b))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Contraction : Binary
  {
    constexpr Contraction(TreeTag tag, node_ptr a, node_ptr b)
        : Binary(tag, std::move(a), std::move(b))
    {
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return a->outer_index() ^ b->outer_index();
    }
  };

  struct Product final : Contraction
  {
    constexpr Product(node_ptr a, node_ptr b)
        : Contraction(PRODUCT, std::move(a), std::move(b))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Ratio final : Contraction
  {
    constexpr Ratio(node_ptr a, node_ptr b)
        : Contraction(RATIO, std::move(a), std::move(b))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Unary : Node
  {
    node_ptr a;

    constexpr Unary(TreeTag tag, node_ptr a)
        : Node(tag)
        , a(std::move(a))
    {
    }

    constexpr auto size() const -> int override
    {
      return a->size() + 1;
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return a->outer_index();
    }
  };

  struct Bind final : Unary
  {
    TensorIndex index;

    constexpr Bind(node_ptr a, TensorIndex i)
        : Unary(BIND, std::move(a))
        , index(i)
    {
      assert(this->a->rank() == index.rank());
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return index;
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Negate final : Unary
  {
    constexpr Negate(node_ptr a)
        : Unary(NEGATE, std::move(a))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Exponent final : Unary
  {
    constexpr Exponent(node_ptr a)
        : Unary(EXPONENT, std::move(a))
    {
      assert(this->a->rank() == 0);
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Partial final : Unary
  {
    TensorIndex index;

    constexpr Partial(node_ptr a, std::same_as<TensorIndex> auto... is)
        : Unary(PARTIAL, std::move(a))
        , index(is...)
    {
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return exclusive(a->outer_index() + index);
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct CMath final : Unary
  {
    CMathTag func;
    Rational q;

    constexpr CMath(node_ptr a, CMathTag f)
        : Unary(CMATH, std::move(a))
        , func(f)
        , q()
    {
    }

    constexpr CMath(node_ptr a, Rational q, CMathTag f)
        : Unary(CMATH, std::move(a))
        , func(f)
        , q(std::move(q))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Leaf : Node
  {
    constexpr Leaf(TreeTag tag)
        : Node(tag)
    {
    }
  };

  struct Literal final : Leaf
  {
    Rational q;

    constexpr Literal(Rational q)
        : Leaf(LITERAL)
        , q(std::move(q))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Delta final : Leaf
  {
    TensorIndex index;

    constexpr Delta(TensorIndex i)
        : Leaf(DELTA)
        , index(i)
    {
      assert(i.size() == 2);
      assert(i.rank() == 2);
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return index;
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

  struct Epsilon final : Leaf
  {
    TensorIndex index;

    constexpr Epsilon(TensorIndex i)
        : Leaf(EPSILON)
        , index(i)
    {
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return index;
    }

    constexpr void destroy() const override
    {
      delete this;
    }
  };

}
