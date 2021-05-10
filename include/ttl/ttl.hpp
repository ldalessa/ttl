#pragma once

#include "ttl/Rational.hpp"
#include "ttl/TensorIndex.hpp"
#include "ttl/cpos.hpp"
#include "ttl/tags.hpp"
#include "ttl/intrusive_ptr.hpp"
#include <ce/dvector.hpp>

namespace ttl
{
  struct LinkedNode
  {
    struct LinkedNode_
    {
      int count = 0;
      constexpr virtual void destroy() const = 0;
      constexpr virtual auto size() const -> int = 0;
      constexpr virtual auto outer_index() const -> TensorIndex = 0;
      constexpr virtual auto tag() const -> TreeTag = 0;
    };

    intrusive_ptr<LinkedNode_> root;

    template <class T>
    constexpr LinkedNode(T t)
        : root(new Box_<T>(std::move(t)))
    {
    }

    template <class T>
    constexpr LinkedNode(Box_<T> box)
        : root(std::move(box))
    {
    }

    constexpr auto size() const -> int
    {
      return root->size();
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return root->outer_index();
    }

    constexpr auto tag() const -> TreeTag
    {
      return root->tag();
    }

    template <class T>
    struct Box_ : LinkedNode_
    {
      T data_;

      constexpr Box_(T data)
          : data_(std::move(data))
      {
      }

      constexpr void destroy() const override
      {
        delete this;
      }

      constexpr auto size() const -> int override
      {
        return ttl::size(this->data_);
      }

      constexpr auto outer_index() const -> TensorIndex override
      {
        return ttl::outer_index(this->data_);
      }

      constexpr auto tag() const -> TreeTag override
      {
        return ttl::tag(this->data_);
      }
    };
  };

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
  };

  template <int N>
  struct LinkedTree
  {
    using linked_tree_tag_t = void;

    LinkedNode root;

    template <class T>
    constexpr LinkedTree(T t)
        : root(std::move(t))
    {
    }

    constexpr auto tag() const -> int
    {
      return root.tag();
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return root.outer_index();
    }

    constexpr LinkedTree<N + 1> operator*() const;
    constexpr LinkedTree<N + 1> operator()(std::same_as<TensorIndex> auto... is) const;
  };

  template <class T>
  concept linked_tree_t = requires {
    typename std::remove_cvref_t<T>::linked_tree_tag_t;
  };

  struct Node
  {
    TreeTag tag_;
    constexpr Node(TreeTag tag)
        : tag_(tag)
    {
    }

    constexpr auto tag() const -> TreeTag
    {
      return tag_;
    }
  };

  struct Binary : Node
  {
    LinkedNode a;
    LinkedNode b;

    constexpr Binary(TreeTag tag, linked_tree_t auto&& a, linked_tree_t auto&& b)
        : Node(tag)
        , a(FWD(a).root)
        , b(FWD(b).root)
    {
    }

    constexpr auto size() const -> int
    {
      return a.size() + b.size() + 1;
    }
  };

  struct Addition : Binary
  {
    constexpr Addition(TreeTag tag, linked_tree_t auto&& a, linked_tree_t auto&& b)
        : Binary(tag, FWD(a), FWD(b))
    {
      assert(permutation(ttl::outer_index(this->a), ttl::outer_index(this->b)));
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return ttl::outer_index(a);
    }
  };

  struct Sum : Addition
  {
    constexpr Sum(linked_tree_t auto&& a, linked_tree_t auto&& b)
        : Addition(SUM, FWD(a), FWD(b))
    {
    }
  };

  struct Difference : Addition
  {
    constexpr Difference(linked_tree_t auto&& a, linked_tree_t auto&& b)
        : Addition(DIFFERENCE, FWD(a), FWD(b))
    {
    }
  };

  struct Contraction : Binary
  {
    constexpr Contraction(TreeTag tag, linked_tree_t auto&& a, linked_tree_t auto&& b)
        : Binary(tag, FWD(a), FWD(b))
    {
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return ttl::outer_index(a) ^ ttl::outer_index(b);
    }
  };

  struct Product : Contraction
  {
    constexpr Product(linked_tree_t auto&& a, linked_tree_t auto&& b)
        : Contraction(PRODUCT, FWD(a), FWD(b))
    {
    }
  };

  struct Ratio : Contraction
  {
    constexpr Ratio(linked_tree_t auto&& a, linked_tree_t auto&& b)
        : Contraction(RATIO, FWD(a), FWD(b))
    {
    }
  };

  struct Unary : Node
  {
    LinkedNode a;

    constexpr Unary(TreeTag tag, linked_tree_t auto&& a)
        : Node(tag)
        , a(FWD(a).root)
    {
    }

    constexpr auto size() const -> int
    {
      return a.size() + 1;
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return ttl::outer_index(a);
    }
  };

  struct Bind : Unary
  {
    TensorIndex index;

    constexpr Bind(linked_tree_t auto&& a, TensorIndex i)
        : Unary(BIND, FWD(a))
        , index(i)
    {
      assert(ttl::rank(this->a) == ttl::rank(index));
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return index;
    }
  };

  struct Negate : Unary
  {
    constexpr Negate(linked_tree_t auto&& a)
        : Unary(NEGATE, FWD(a))
    {
    }
  };

  struct Exponent : Unary
  {
    constexpr Exponent(linked_tree_t auto&& a)
        : Unary(EXPONENT, FWD(a))
    {
      assert(ttl::rank(this->a) == 0);
    }
  };

  struct Partial : Unary
  {
    TensorIndex index;

    constexpr Partial(linked_tree_t auto&& a, std::same_as<TensorIndex> auto... is)
        : Unary(PARTIAL, FWD(a))
        , index(is...)
    {
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return exclusive(ttl::outer_index(a) + index);
    }
  };

  struct CMath : Unary
  {
    CMathTag func;
    Rational q;

    constexpr CMath(linked_tree_t auto&& a, CMathTag f)
        : Unary(CMATH, FWD(a))
        , func(f)
        , q()
    {
    }

    constexpr CMath(linked_tree_t auto&& a, Rational q, CMathTag f)
        : Unary(CMATH, FWD(a))
        , func(f)
        , q(std::move(q))
    {
    }
  };

  struct Leaf : Node
  {
    constexpr Leaf(TreeTag tag)
        : Node(tag)
    {
    }

    constexpr auto size() const -> int
    {
      return 1;
    }
  };

  struct Delta : Leaf
  {
    TensorIndex index;

    constexpr Delta(TensorIndex i)
        : Leaf(DELTA)
        , index(i)
    {
      assert(ttl::size(i) == 2);
      assert(ttl::rank(i) == 2);
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return index;
    }
  };

  struct Epsilon : Leaf
  {
    TensorIndex index;

    constexpr Epsilon(TensorIndex i)
        : Leaf(EPSILON)
        , index(i)
    {
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return index;
    }
  };

  constexpr auto promote(linked_tree_t auto&& a) -> decltype(auto)
  {
    return FWD(a);
  }

  constexpr auto promote(Rational q) -> LinkedTree<1>
  {
    return { q };
  }

  constexpr auto promote(std::integral auto i) -> LinkedTree<1>
  {
    return { i };
  }

  constexpr auto promote(std::floating_point auto d) -> LinkedTree<1>
  {
    return { d };
  }

  template <int A, int B>
  constexpr auto operator+(LinkedTree<A> a, LinkedTree<B> b) -> LinkedTree<A + B + 1>
  {
    return { Sum(std::move(a), std::move(b)) };
  }

  template <int A>
  constexpr auto operator+(LinkedTree<A> a) -> LinkedTree<A>
  {
    return std::move(a);
  }

  template <int A, int B>
  constexpr auto operator-(LinkedTree<A> a, LinkedTree<B> b) -> LinkedTree<A + B + 1>
  {
    return { Difference(std::move(a), std::move(b)) };
  }

  template <int A>
  constexpr auto operator-(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { Negate(std::move(a)) };
  }

  template <int A, int B>
  constexpr auto operator*(LinkedTree<A> a, LinkedTree<B> b) -> LinkedTree<A + B + 1>
  {
    return { Product(std::move(a), std::move(b)) };
  }

  template <int A, int B>
  constexpr auto operator/(LinkedTree<A> a, LinkedTree<B> b) -> LinkedTree<A + B + 1>
  {
    return { Ratio(std::move(a), std::move(b)) };
  }

  template <int A>
  constexpr auto LinkedTree<A>::operator*() const -> LinkedTree<A + 1>
  {
    return { Exponent(*this) };
  }

  template <int A>
  constexpr auto LinkedTree<A>::operator()(std::same_as<TensorIndex> auto... is) const -> LinkedTree<A + 1>
  {
    return { Bind(*this, (is + ...)) };
  }

  template <int A>
  constexpr auto D(LinkedTree<A> a, std::same_as<TensorIndex> auto... is) -> LinkedTree<A + 1>
  {
    return { Partial(std::move(a), is...) };
  }

  constexpr auto δ(TensorIndex i, TensorIndex j) -> LinkedTree<1>
  {
    return { Delta(i + j) };
  }

  constexpr auto ε(std::same_as<TensorIndex> auto... is) -> LinkedTree<1>
  {
    return { Epsilon((is + ...)) };
  }

  template <int A>
  constexpr auto abs(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    assert(ttl::rank(a) == 0);
    return { CMath(std::move(a), ABS) };
  }

  template <int A>
  constexpr auto fmin(LinkedTree<A> a, Rational q) -> LinkedTree<A + 1>
  {
    assert(ttl::rank(a) == 0);
    return { CMath(std::move(a), std::move(q), FMIN) };
  }

  template <int A>
  constexpr auto fmax(LinkedTree<A> a, Rational q) -> LinkedTree<A + 1>
  {
    assert(ttl::rank(a) == 0);
    return { CMath(std::move(a), std::move(q), FMAX) };
  }

  template <int A>
  constexpr auto exp(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), EXP) };
  }

  template <int A>
  constexpr auto log(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), LOG) };
  }

  template <int A>
  constexpr auto pow(LinkedTree<A> a, Rational q) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), std::move(q), LOG) };
  }

  template <int A>
  constexpr auto sqrt(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), SQRT) };
  }

  template <int A>
  constexpr auto sin(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), SIN) };
  }

  template <int A>
  constexpr auto cos(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), COS) };
  }

  template <int A>
  constexpr auto tan(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), TAN) };
  }

  template <int A>
  constexpr auto asin(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), ASIN) };
  }

  template <int A>
  constexpr auto acos(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), ACOS) };
  }

  template <int A>
  constexpr auto atan(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), ATAN) };
  }

  template <int A>
  constexpr auto atan2(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), ATAN2) };
  }

  template <int A>
  constexpr auto sinh(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), SINH) };
  }

  template <int A>
  constexpr auto cosh(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), COSH) };
  }

  template <int A>
  constexpr auto tanh(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), TANH) };
  }

  template <int A>
  constexpr auto asinh(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), ASINH) };
  }

  template <int A>
  constexpr auto acosh(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), ACOSH) };
  }

  template <int A>
  constexpr auto atanh(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), ATANH) };
  }

  template <int A>
  constexpr auto ceil(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), CEIL) };
  }

  template <int A>
  constexpr auto floor(LinkedTree<A> a) -> LinkedTree<A + 1>
  {
    return { CMath(std::move(a), FLOOR) };
  }
}
