#pragma once

#include "ttl/Rational.hpp"
#include "ttl/Tags.hpp"
#include "ttl/TensorIndex.hpp"
#include "ttl/TensorRef.hpp"
#include "ttl/cpos.hpp"
#include <memory>
#include <raberu.hpp>
#include <string_view>

#ifndef FWD
#define FWD(a) std::forward<decltype(a)>(a)
#endif

namespace ttl::prop
{
  using namespace rbr::literals;
  inline constexpr auto id = "id"_kw;
}

namespace ttl::nodes
{
  struct node_ptr;

  struct Node
  {
    using node_tag = void;

    int count = 0;
    std::string_view id_;

    constexpr virtual void destroy() const = 0;

    constexpr virtual auto tag() const -> TreeTag = 0;

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

    constexpr virtual void print(fmt::memory_buffer&) const = 0;

    constexpr auto id() const -> std::string_view
    {
      return id_;
    }

    constexpr Node& operator[](rbr::keyword_parameter auto param)
    {
      rbr::settings args = { param };
      id_ = args[prop::id];
      return *this;
    }
  };

  template <class T>
  concept node_t = requires {
    typename std::remove_cvref_t<T>::node_tag;
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

    constexpr node_ptr& operator=(Node* b)
    {
      if (b) {
        ++b->count;
      }
      dec();
      ptr_ = b;
      return *this;
    }

    constexpr friend auto operator<=>(node_ptr const&, node_ptr const&) = default;

    constexpr auto operator->() const -> Node*
    {
      return ptr_;
    }

    constexpr auto operator*() const -> Node&
    {
      return *ptr_;
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
    using binary_node_tag = void;

    node_ptr a;
    node_ptr b;

    constexpr Binary(node_ptr a, node_ptr b)
        : a(std::move(a))
        , b(std::move(b))
    {
      id_ = this->b->id();
    }

    constexpr auto size() const -> int override
    {
      return a->size() + b->size() + 1;
    }

    void dot(fmt::memory_buffer& out, int i, int b, int a) const
    {
      auto outer = outer_index();
      if (outer.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{} ↑{}\"]\n", i, tag(), outer);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, tag(), outer);
      }
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
      fmt::format_to(out, "\tnode{} -- node{}\n", i, b);
    }

    void print(fmt::memory_buffer& out) const override
    {
      out.append(std::string_view("("));
      a->print(out);
      fmt::format_to(out, " {} ", tag());
      b->print(out);
      out.append(std::string_view(")"));
    }
  };

  template <class T>
  concept binary_node_t = node_t<T> and requires {
    typename std::remove_cvref_t<T>::binary_node_tag;
  };

  struct Addition : Binary
  {
    constexpr Addition(node_ptr a, node_ptr b)
        : Binary(std::move(a), std::move(b))
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
        : Addition(std::move(a), std::move(b))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return SUM;
    }
  };

  struct Difference final : Addition
  {
    constexpr Difference(node_ptr a, node_ptr b)
        : Addition(std::move(a), std::move(b))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return DIFFERENCE;
    }
  };

  struct Contraction : Binary
  {
    constexpr Contraction(node_ptr a, node_ptr b)
        : Binary(std::move(a), std::move(b))
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
        : Contraction(std::move(a), std::move(b))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return PRODUCT;
    }
  };

  struct Ratio final : Contraction
  {
    constexpr Ratio(node_ptr a, node_ptr b)
        : Contraction(std::move(a), std::move(b))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return RATIO;
    }
  };

  struct Unary : Node
  {
    using unary_node_tag = void;

    node_ptr a;

    constexpr Unary(node_ptr a)
        : a(std::move(a))
    {
      id_ = this->a->id();
    }

    constexpr auto size() const -> int override
    {
      return a->size() + 1;
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return a->outer_index();
    }

    void print(fmt::memory_buffer& out) const override
    {
      fmt::format_to(out, "{}(", tag());
      a->print(out);
      out.append(std::string_view(")"));
    }

    virtual void dot(fmt::memory_buffer& out, int i, int a) const
    {
      fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, tag());
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
    }
  };

  template <class T>
  concept unary_node_t = node_t<T> and requires {
    typename std::remove_cvref_t<T>::unary_node_tag;
  };

  struct Binder : Unary
  {
    TensorIndex index;

    constexpr Binder(node_ptr a, TensorIndex i)
        : Unary(std::move(a))
        , index(i)
    {
    }

    void print(fmt::memory_buffer& out) const override
    {
      fmt::format_to(out, "{}(", tag());
      a->print(out);
      fmt::format_to(out, ",{})", index);
    }

    void dot(fmt::memory_buffer& out, int i, int a) const override
    {
      TensorIndex outer = outer_index();
      TensorIndex child = this->a->outer_index();
      if (outer.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{}({},{}) ↑{}\"]\n", i, tag(), child, index, outer);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}({},{})\"]\n", i, tag(), child, index);
      }
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
    }
  };

  struct Bind final : Binder
  {
    constexpr Bind(node_ptr a, TensorIndex index)
        : Binder(std::move(a), index)
    {
      assert(this->a->rank() == index.rank());
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return BIND;
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return exclusive(index);
    }
  };

  struct Partial final : Binder
  {
    constexpr Partial(node_ptr a, TensorIndex index)
        : Binder(std::move(a), index)
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return PARTIAL;
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return exclusive(a->outer_index() + index);
    }
  };

  struct Negate final : Unary
  {
    constexpr Negate(node_ptr a)
        : Unary(std::move(a))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return NEGATE;
    }
  };

  struct Exponent final : Unary
  {
    constexpr Exponent(node_ptr a)
        : Unary(std::move(a))
    {
      assert(this->a->rank() == 0);
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return EXPONENT;
    }
  };

  struct CMath final : Unary
  {
    CMathTag func;
    Rational q;

    constexpr CMath(node_ptr a, CMathTag f)
        : Unary(std::move(a))
        , func(f)
        , q(0)
    {
    }

    constexpr CMath(node_ptr a, Rational q, CMathTag f)
        : Unary(std::move(a))
        , func(f)
        , q(std::move(q))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return CMATH;
    }

    void print(fmt::memory_buffer& out) const override
    {
      fmt::format_to(out, "{}(", func);
      a->print(out);
      out.append(std::string_view(")"));
    }

    void dot(fmt::memory_buffer& out, int i, int a) const override
    {
      TensorIndex index = outer_index();
      if (index.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{} ↑{}\"]\n", i, func, index);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, func);
      }
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
    }
  };

  struct Leaf : Node
  {
    using leaf_node_tag = void;

    virtual void dot(fmt::memory_buffer& out, int i) const = 0;
  };

  template <class T>
  concept leaf_node_t = node_t<T> and requires {
    typename std::remove_cvref_t<T>::leaf_node_tag;
  };

  struct Tensor final : Leaf
  {
    ttl::TensorRef const *a;

    constexpr Tensor(ttl::TensorRef const *a)
        : a(a)
    {
      id_ = a->id();
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto rank() const -> int override
    {
      return a->rank();
    }

    constexpr auto tag() const -> TreeTag override
    {
      return TENSOR;
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return a->outer_index();
    }

    void print(fmt::memory_buffer& out) const override
    {
      if (id().size() == 0) {
        a->print(out);
      }
      else {
        fmt::format_to(out, "{}", id());
      }
    }

    void dot(fmt::memory_buffer& out, int i) const override
    {
      fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, a->id());
    }
  };

  struct Literal final : Leaf
  {
    Rational q;

    constexpr Literal(Rational q)
        : q(std::move(q))
    {
    }

    constexpr void destroy() const override
    {
      delete this;
    }

    constexpr auto tag() const -> TreeTag override
    {
      return LITERAL;
    }

    void dot(fmt::memory_buffer& out, int i) const override
    {
      fmt::format_to(out, "\tnode{}[label=\"{}()\"]\n", i, tag());
    }

    void print(fmt::memory_buffer& out) const override
    {
      fmt::format_to(out, "{}", q);
    }
  };

  struct Delta final : Leaf
  {
    TensorIndex index;

    constexpr Delta(TensorIndex i)
        : index(i)
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

    constexpr auto tag() const -> TreeTag override
    {
      return DELTA;
    }

    void dot(fmt::memory_buffer& out, int i) const override
    {
      fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, tag(), index);
    }

    void print(fmt::memory_buffer& out) const override
    {
      fmt::format_to(out, "{}({})", tag(), index);
    }
  };

  struct Epsilon final : Leaf
  {
    TensorIndex index;

    constexpr Epsilon(TensorIndex i)
        : index(i)
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

    constexpr auto tag() const -> TreeTag override
    {
      return EPSILON;
    }

    void dot(fmt::memory_buffer& out, int i) const override
    {
      fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, tag(), index);
    }

    void print(fmt::memory_buffer& out) const override
    {
      fmt::format_to(out, "{}({})", tag(), index);
    }
  };

  // helper type for the visitor #4
  template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
  // explicit deduction guide (not needed as of C++20)
  template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

  // constexpr auto tag_invoke(visit_tag_, node_t auto const& n, auto&& op)
  constexpr auto tag_invoke(visit_tag_, Node const& node, auto&& op)
  {
    switch (node.tag())
    {
     case SUM:        return FWD(op)(static_cast<Sum const&>(node));
     case DIFFERENCE: return FWD(op)(static_cast<Difference const&>(node));
     case PRODUCT:    return FWD(op)(static_cast<Product const&>(node));
     case RATIO:      return FWD(op)(static_cast<Ratio const&>(node));
     case BIND:       return FWD(op)(static_cast<Bind const&>(node));
     case NEGATE:     return FWD(op)(static_cast<Negate const&>(node));
     case EXPONENT:   return FWD(op)(static_cast<Exponent const&>(node));
     case PARTIAL:    return FWD(op)(static_cast<Partial const&>(node));
     case CMATH:      return FWD(op)(static_cast<CMath const&>(node));
     case LITERAL:    return FWD(op)(static_cast<Literal const&>(node));
     case TENSOR:     return FWD(op)(static_cast<Tensor const&>(node));
     case SCALAR:     abort();
     case DELTA:      return FWD(op)(static_cast<Delta const&>(node));
     case EPSILON:    return FWD(op)(static_cast<Epsilon const&>(node));
    }
    __builtin_unreachable();
  }

  struct AnyNode final : Node
  {
    using any_node_tag = void;

    TreeTag tag_;
    union {
      Sum sum;
      Difference difference;
      Product product;
      Ratio ratio;
      Bind bind;
      Negate negate;
      Exponent exponent;
      Partial partial;
      CMath cmath;
      Tensor tensor;
      Literal literal;
      Delta delta;
      Epsilon epsilon;
    };

    constexpr AnyNode() {}
    constexpr ~AnyNode()
    {
    }

    constexpr AnyNode& operator=(Node const& b)
    {
      visit(b, [&](auto const& b) { *this = b; });
      return *this;
    }

    constexpr auto operator=(Sum const& b) -> Sum&
    {
      tag_ = SUM;
      return *std::construct_at(&sum, b);
    }

    constexpr auto operator=(Difference const& b) -> Difference&
    {
      tag_ = DIFFERENCE;
      return *std::construct_at(&difference, b);
    }

    constexpr auto operator=(Product const& b) -> Product&
    {
      tag_ = PRODUCT;
      return *std::construct_at(&product, b);
    }

    constexpr auto operator=(Ratio const& b) -> Ratio&
    {
      tag_ = RATIO;
      return *std::construct_at(&ratio, b);
    }

    constexpr auto operator=(Bind const& b) -> Bind&
    {
      tag_ = BIND;
      return *std::construct_at(&bind, b);
    }

    constexpr auto operator=(Negate const& b) -> Negate&
    {
      tag_ = NEGATE;
      return *std::construct_at(&negate, b);
    }

    constexpr auto operator=(Exponent const& b) -> Exponent&
    {
      tag_ = EXPONENT;
      return *std::construct_at(&exponent, b);
    }

    constexpr auto operator=(Partial const& b) -> Partial&
    {
      tag_ = PARTIAL;
      return *std::construct_at(&partial, b);
    }

    constexpr auto operator=(CMath const& b) -> CMath&
    {
      tag_ = CMATH;
      return *std::construct_at(&cmath, b);
    }

    constexpr auto operator=(Literal const& b) -> Literal&
    {
      tag_ = LITERAL;
      return *std::construct_at(&literal, b);
    }

    constexpr auto operator=(Tensor const& b) -> Tensor&
    {
      tag_ = TENSOR;
      return *std::construct_at(&tensor, b);
    }

    constexpr auto operator=(Delta const& b) -> Delta&
    {
      tag_ = DELTA;
      return *std::construct_at(&delta, b);
    }

    constexpr auto operator=(Epsilon const& b) -> Epsilon&
    {
      tag_ = EPSILON;
      return *std::construct_at(&epsilon, b);
    }

    constexpr friend auto tag_invoke(visit_tag_, AnyNode const& n, auto&& op)
    {
      switch (n.tag_)
      {
       case SUM:        return FWD(op)(n.sum);
       case DIFFERENCE: return FWD(op)(n.difference);
       case PRODUCT:    return FWD(op)(n.product);
       case RATIO:      return FWD(op)(n.ratio);
       case BIND:       return FWD(op)(n.bind);
       case NEGATE:     return FWD(op)(n.negate);
       case EXPONENT:   return FWD(op)(n.exponent);
       case PARTIAL:    return FWD(op)(n.partial);
       case CMATH:      return FWD(op)(n.cmath);
       case LITERAL:    return FWD(op)(n.literal);
       case TENSOR:     return FWD(op)(n.tensor);
       case SCALAR:     abort();
       case DELTA:      return FWD(op)(n.delta);
       case EPSILON:    return FWD(op)(n.epsilon);
      }
      __builtin_unreachable();
    }

    constexpr void destroy() const override
    {
    }

    constexpr auto tag() const -> TreeTag override
    {
      return tag_;
    }

    constexpr auto size() const -> int override
    {
      return visit(*this, [&](node_t auto&& b) { return FWD(b).size(); });
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return visit(*this, [&](node_t auto&& b) { return FWD(b).outer_index(); });
    }

    constexpr auto rank() const -> int override
    {
      return visit(*this, [&](node_t auto&& b) { return FWD(b).rank(); });
    }

    void print(fmt::memory_buffer& out) const override
    {
      return visit(*this, [&](node_t auto&& b) { return FWD(b).print(out); });
    }
  };

  template <class T>
  concept any_node_t = node_t<T> and requires {
    typename std::remove_cvref_t<T>::any_node_tag;
  };

  static_assert(node_t<Sum const&> and binary_node_t<Sum const&> and not
  unary_node_t<Sum const&> and not leaf_node_t<Sum const&>);
}
