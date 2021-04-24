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

  struct Node
  {
    Tag   tag = {};
    int count = 0;

    constexpr Node() = default;

    constexpr Node(Tag tag)
        : tag(tag)
    {
    }

    virtual auto to_string() const -> std::string = 0;

    friend auto to_string(Node const& tree) -> std::string
    {
      return tree.to_string();
    }
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
      dec();
      ptr_ = b.ptr_;
      inc();
      return *this;
    }

    constexpr node_ptr& operator=(node_ptr&& b)
    {
      dec();
      ptr_ = std::exchange(b.ptr_, nullptr);
      return *this;
    }


    constexpr node_ptr& operator=(std::nullptr_t)
    {
      dec();
      return *this;
    }

    constexpr explicit operator bool() const
    {
      return ptr_ != nullptr;
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
    }

    auto to_string() const -> std::string override
    {
      return fmt::format("({} {} {})", a->to_string(), tag, b->to_string());
    }
  };

  struct Unary : Node
  {
    node_ptr b;

    constexpr Unary(Tag tag, rbr::keyword_parameter auto... params)
        : Node(tag)
    {
      rbr::settings args = { params... };
      b = args[kw::b];
      assert(b);
    }
  };

  struct Leaf : Node
  {
    constexpr Leaf(Tag tag, rbr::keyword_parameter auto... params)
        : Node(tag)
    {
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

    auto to_string() const -> std::string override
    {
      return fmt::format("{}({},{})", tag, b->to_string(), index);
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

    auto to_string() const -> std::string override
    {
      return fmt::format("{}({},{})", tag, b->to_string(), index);
    }
  };

  struct Sqrt : Unary
  {
    constexpr Sqrt(rbr::keyword_parameter auto... params)
        : Unary(SQRT, params...)
    {
    }

    auto to_string() const -> std::string override
    {
      return fmt::format("{}({})", tag, b->to_string());
    }
  };

  struct Exp : Unary
  {
    constexpr Exp(rbr::keyword_parameter auto... params)
        : Unary(EXP, params...)
    {
    }

    auto to_string() const -> std::string override
    {
      return fmt::format("{}({})", tag, b->to_string());
    }
  };

  struct Negate : Unary
  {
    constexpr Negate(rbr::keyword_parameter auto... params)
        : Unary(NEGATE, params...)
    {
    }

    auto to_string() const -> std::string override
    {
      return fmt::format("{}({})", tag, b->to_string());
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

    auto to_string() const -> std::string override
    {
      return fmt::format("{}", q);
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

    auto to_string() const -> std::string override
    {
      return fmt::format("{}", d);
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

    auto to_string() const -> std::string override
    {
      if (index.size()) {
        return fmt::format("{}({})", *tensor, index);
      }
      else {
        return fmt::format("{}", *tensor);
      }
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

    auto to_string() const -> std::string override
    {
      if (index.size()) {
        return fmt::format("{}({})", *tensor, index);
      }
      else {
        return fmt::format("{}", *tensor);
      }
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

    auto to_string() const -> std::string override
    {
      return fmt::format("{}({})", tag, index);
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

    auto to_string() const -> std::string override
    {
      return fmt::format("{}({})", tag, index);
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

  constexpr node_ptr::node_ptr(Tag tag, rbr::keyword_parameter auto... params)
  {
    switch (tag)
    {
     case SUM:        ptr_ = new Sum(params...); break;
     case DIFFERENCE: ptr_ = new Difference(params...); break;
     case PRODUCT:    ptr_ = new Product(params...); break;
     case RATIO:      ptr_ = new Ratio(params...); break;
     case BIND:       ptr_ = new Bind(params...); break;
     case PARTIAL:    ptr_ = new Partial(params...); break;
     case POW:        ptr_ = new Pow(params...); break;
     case SQRT:       ptr_ = new Sqrt(params...); break;
     case EXP:        ptr_ = new Exp(params...); break;
     case NEGATE:     ptr_ = new Negate(params...); break;
     case RATIONAL:   ptr_ = new Rational(params...); break;
     case DOUBLE:     ptr_ = new Double(params...); break;
     case TENSOR:     ptr_ = new Tensor(params...); break;
     case SCALAR:     ptr_ = new Scalar(params...); break;
     case DELTA:      ptr_ = new Delta(params...); break;
     case EPSILON:    ptr_ = new Epsilon(params...); break;
     default:
      assert(false);
      __builtin_unreachable();
    }
    inc();
  }
}