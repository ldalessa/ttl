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

  struct node_ptr;

  struct Node
  {
    Node* parent = nullptr;
    Tag      tag = {};
    int    count = 0;

    constexpr Node() = default;

    constexpr Node(Tag tag)
        : tag(tag)
    {
    }

    constexpr virtual void replace(Node*, node_ptr&) {}
    constexpr virtual auto outer() const -> Index = 0;
  };

  struct node_ptr
  {
    Node *ptr_ = nullptr;

    constexpr ~node_ptr()
    {
      dec();
    }

    constexpr node_ptr() = default;

    constexpr node_ptr(Tag, rbr::keyword_parameter auto...);

    constexpr node_ptr(node_ptr const& b)
        : ptr_(b.ptr_)
    {
      inc();
    }

    constexpr node_ptr(node_ptr&& b)
        : ptr_(std::exchange(b.ptr_, nullptr))
    {
    }

    constexpr node_ptr& operator=(node_ptr const& b)
    {
      assert(*this != b);
      dec();
      ptr_ = b.ptr_;
      inc();
      return *this;
    }

    constexpr node_ptr& operator=(node_ptr&& b)
    {
      assert(b != *this);
      dec();
      ptr_ = std::exchange(b.ptr_, nullptr);
      return *this;
    }

    constexpr node_ptr& operator=(Node* ptr)
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

    constexpr friend bool operator==(node_ptr const& a, Node* b) {
      return a.ptr_ == b;
    }

    constexpr friend bool operator==(Node* a, node_ptr const& b) {
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

    constexpr auto visit(auto&& op, auto&&... args) const;

    constexpr void inc() const;
    constexpr void dec();
  };

  struct Binary : Node
  {
    node_ptr a;
    node_ptr b;

    constexpr Binary(Tag tag, rbr::keyword_parameter auto... params)
        : Node(tag)
    {
      rbr::settings args = { params... };
      a = args[kw::a];
      b = args[kw::b];
      assert(a);
      assert(b);
      a->parent = this;
      b->parent = this;
    }

    constexpr void replace(Node* child, node_ptr& with) override
    {
      assert(child);
      assert(a == child or b == child);
      child->parent = nullptr;
      if (a == child) a = with;
      else b = with;
      with->parent = this;
    }

    constexpr auto outer() const -> Index override
    {
      return tag.outer(a->outer(), b->outer());
    }
  };

  struct Unary : Node
  {
    node_ptr a;

    constexpr Unary(Tag tag, rbr::keyword_parameter auto... params)
        : Node(tag)
    {
      rbr::settings args = { params... };
      a = args[kw::a];
      assert(a);
      a->parent = this;
    }

    constexpr void replace(Node* child, node_ptr& with) override
    {
      assert(child && a == child);
      child->parent = nullptr;
      a = with;
      with->parent = this;
    }

    constexpr auto outer() const -> Index override
    {
      return a->outer();
    }
  };

  struct Leaf : Node
  {
    constexpr Leaf(Tag tag, rbr::keyword_parameter auto... params)
        : Node(tag)
    {
    }

    constexpr auto outer() const -> Index override
    {
      return {};
    }
  };

  struct Sum : Binary
  {
    constexpr Sum(rbr::keyword_parameter auto... params)
        : Binary(SUM, params...)
    {
    }
  };

  struct Difference : Binary
  {
    constexpr Difference(rbr::keyword_parameter auto... params)
        : Binary(DIFFERENCE, params...)
    {
    }
  };

  struct Product : Binary
  {
    constexpr Product(rbr::keyword_parameter auto... params)
        : Binary(PRODUCT, params...)
    {
    }
  };

  struct Ratio : Binary
  {
    constexpr Ratio(rbr::keyword_parameter auto... params)
        : Binary(RATIO, params...)
    {
    }
  };

  struct Pow : Binary
  {
    constexpr Pow(rbr::keyword_parameter auto... params)
        : Binary(POW, params...)
    {
    }
  };

  struct Bind : Unary
  {
    Index index;

    constexpr Bind(rbr::keyword_parameter auto... params)
        : Unary(BIND, params...)
    {
      rbr::settings args = { params... };
      index = args[kw::tensor_index];
    }

    constexpr auto outer() const -> Index override
    {
      return tag.outer(a->outer(), index);
    }
  };

  struct Partial : Unary
  {
    Index index;

    constexpr Partial(rbr::keyword_parameter auto... params)
        : Unary(PARTIAL, params...)
    {
      rbr::settings args = { params... };
      index = args[kw::tensor_index];
    }

    constexpr auto outer() const -> Index override
    {
      return tag.outer(a->outer(), index);
    }
  };

  struct Sqrt : Unary
  {
    constexpr Sqrt(rbr::keyword_parameter auto... params)
        : Unary(SQRT, params...)
    {
    }
  };

  struct Exp : Unary
  {
    constexpr Exp(rbr::keyword_parameter auto... params)
        : Unary(EXP, params...)
    {
    }
  };

  struct Negate : Unary
  {
    constexpr Negate(rbr::keyword_parameter auto... params)
        : Unary(NEGATE, params...)
    {
    }
  };

  struct Rational : Leaf
  {
    ttl::Rational q;

    constexpr Rational(rbr::keyword_parameter auto... params)
        : Leaf(RATIONAL, params...)
    {
      rbr::settings args = { params... };
      q = args[kw::q];
    }
  };

  struct Double : Leaf
  {
    double d;

    constexpr Double(rbr::keyword_parameter auto... params)
        : Leaf(DOUBLE, params...)
    {
      rbr::settings args = { params... };
      d = args[kw::d];
    }
  };

  struct Tensor : Leaf
  {
    ttl::Tensor const* tensor;
    Index index;

    constexpr Tensor(rbr::keyword_parameter auto... params)
        : Leaf(TENSOR, params...)
    {
      rbr::settings args = { params... };
      tensor = args[kw::tensor];
      index = args[kw::tensor_index];
    }

    constexpr auto outer() const -> Index override
    {
      return tag.outer(index);
    }
  };

  struct Scalar : Leaf
  {
    ttl::Tensor const* tensor;
    ScalarIndex index;

    constexpr Scalar(rbr::keyword_parameter auto... params)
        : Leaf(SCALAR, params...)
    {
      rbr::settings args = { params... };
      tensor = args[kw::tensor];
      index = args[kw::scalar_index];
    }
  };

  struct Delta : Leaf
  {
    Index index;

    constexpr Delta(rbr::keyword_parameter auto... params)
        : Leaf(DELTA, params...)
    {
      rbr::settings args = { params... };
      index = args[kw::tensor_index];
    }

    constexpr auto outer() const -> Index override
    {
      return tag.outer(index);
    }
  };

  struct Epsilon : Leaf
  {
    Index index;

    constexpr Epsilon(rbr::keyword_parameter auto... params)
        : Leaf(EPSILON, params...)
    {
      rbr::settings args = { params... };
      index = args[kw::tensor_index];
    }

    constexpr auto outer() const -> Index override
    {
      return tag.outer(index);
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
        switch (ptr_->tag)
        {
         case SUM:        delete static_cast<Sum*>(ptr_);        break;
         case DIFFERENCE: delete static_cast<Difference*>(ptr_); break;
         case PRODUCT:    delete static_cast<Product*>(ptr_);    break;
         case RATIO:      delete static_cast<Ratio*>(ptr_);      break;
         case BIND:       delete static_cast<Bind*>(ptr_);       break;
         case PARTIAL:    delete static_cast<Partial*>(ptr_);    break;
         case POW:        delete static_cast<Pow*>(ptr_);        break;
         case SQRT:       delete static_cast<Sqrt*>(ptr_);       break;
         case EXP:        delete static_cast<Exp*>(ptr_);        break;
         case NEGATE:     delete static_cast<Negate*>(ptr_);     break;
         case RATIONAL:   delete static_cast<Rational*>(ptr_);   break;
         case DOUBLE:     delete static_cast<Double*>(ptr_);     break;
         case TENSOR:     delete static_cast<Tensor*>(ptr_);     break;
         case SCALAR:     delete static_cast<Scalar*>(ptr_);     break;
         case DELTA:      delete static_cast<Delta*>(ptr_);      break;
         case EPSILON:    delete static_cast<Epsilon*>(ptr_);    break;
         default:
          assert(false);
          __builtin_unreachable();
        }
        ptr_ = nullptr;
      }
    }
  }

  constexpr auto node_ptr::visit(auto&& op, auto&&... args) const
  {
    switch (ptr_->tag)
    {
     case SUM:        return op(static_cast<Sum*>(ptr_), std::forward<decltype(args)>(args)...);
     case DIFFERENCE: return op(static_cast<Difference*>(ptr_), std::forward<decltype(args)>(args)...);
     case PRODUCT:    return op(static_cast<Product*>(ptr_), std::forward<decltype(args)>(args)...);
     case RATIO:      return op(static_cast<Ratio*>(ptr_), std::forward<decltype(args)>(args)...);
     case BIND:       return op(static_cast<Bind*>(ptr_), std::forward<decltype(args)>(args)...);
     case PARTIAL:    return op(static_cast<Partial*>(ptr_), std::forward<decltype(args)>(args)...);
     case POW:        return op(static_cast<Pow*>(ptr_), std::forward<decltype(args)>(args)...);
     case SQRT:       return op(static_cast<Sqrt*>(ptr_), std::forward<decltype(args)>(args)...);
     case EXP:        return op(static_cast<Exp*>(ptr_), std::forward<decltype(args)>(args)...);
     case NEGATE:     return op(static_cast<Negate*>(ptr_), std::forward<decltype(args)>(args)...);
     case RATIONAL:   return op(static_cast<Rational*>(ptr_), std::forward<decltype(args)>(args)...);
     case DOUBLE:     return op(static_cast<Double*>(ptr_), std::forward<decltype(args)>(args)...);
     case TENSOR:     return op(static_cast<Tensor*>(ptr_), std::forward<decltype(args)>(args)...);
     case SCALAR:     return op(static_cast<Scalar*>(ptr_), std::forward<decltype(args)>(args)...);
     case DELTA:      return op(static_cast<Delta*>(ptr_), std::forward<decltype(args)>(args)...);
     case EPSILON:    return op(static_cast<Epsilon*>(ptr_), std::forward<decltype(args)>(args)...);
     default:
      assert(false);
      __builtin_unreachable();
    }
  }

  constexpr node_ptr::node_ptr(Tag tag, rbr::keyword_parameter auto... params)
  {
    switch (tag)
    {
     case SUM:        ptr_ = new Sum(params...);        break;
     case DIFFERENCE: ptr_ = new Difference(params...); break;
     case PRODUCT:    ptr_ = new Product(params...);    break;
     case RATIO:      ptr_ = new Ratio(params...);      break;
     case BIND:       ptr_ = new Bind(params...);       break;
     case PARTIAL:    ptr_ = new Partial(params...);    break;
     case POW:        ptr_ = new Pow(params...);        break;
     case SQRT:       ptr_ = new Sqrt(params...);       break;
     case EXP:        ptr_ = new Exp(params...);        break;
     case NEGATE:     ptr_ = new Negate(params...);     break;
     case RATIONAL:   ptr_ = new Rational(params...);   break;
     case DOUBLE:     ptr_ = new Double(params...);     break;
     case TENSOR:     ptr_ = new Tensor(params...);     break;
     case SCALAR:     ptr_ = new Scalar(params...);     break;
     case DELTA:      ptr_ = new Delta(params...);      break;
     case EPSILON:    ptr_ = new Epsilon(params...);    break;
     default:
      assert(false);
      __builtin_unreachable();
    }
    inc();
  }
}
