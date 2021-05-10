#pragma once

#include "ttl/Nodes.hpp"
#include "ttl/Rational.hpp"
#include "ttl/TensorIndex.hpp"
#include <concepts>

#ifndef FWD
#define FWD(a) std::forward<decltype(a)>(a)
#endif

namespace ttl
{
  template <int A>
  struct Tree
  {
    using tree_tag_t = void;

    node_ptr root;

    constexpr Tree(Node* ptr)
        : root(ptr)
    {
    }

    constexpr auto tag() const -> int
    {
      return root->tag();
    }

    constexpr auto size() const -> int
    {
      return A;
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return root->outer_index();
    }

    constexpr auto rank() const -> int
    {
      return root->rank();
    }

    constexpr auto operator*() const
      -> Tree<A + 1>
    {
      return { new Exponent(root) }; // copy root
    }

    constexpr auto operator()(std::same_as<TensorIndex> auto... is) const
      -> Tree<A + 1>
    {
      return { new Bind(root, (is + ...)) }; // copy root
    }
  };

  template <class T>
  concept tree_t = requires {
    typename std::remove_cvref_t<T>::tree_tag_t;
  };

  template <class T>
  concept expression_t =
  tree_t<T> ||
  std::same_as<std::remove_cvref_t<T>, Rational> ||
  std::integral<T> ||
  std::floating_point<T>;

  template <expression_t... Exprs>
  constexpr inline int tree_size_v = (tree_size_v<Exprs> + ... + 1);

  template <expression_t Expr>
  constexpr inline int tree_size_v<Expr> = 1;

  template <int A>
  constexpr inline int tree_size_v<Tree<A>> = A;

  constexpr auto promote(tree_t auto&& a) -> node_ptr
  {
    return FWD(a).root;
  }

  constexpr auto promote(Rational q) -> node_ptr
  {
    return { new Literal(q) };
  }

  constexpr auto promote(std::integral auto i) -> node_ptr
  {
    return { new Literal(i) };
  }

  constexpr auto promote(std::floating_point auto d) -> node_ptr
  {
    return { new Literal(int(d)) };
  }

  template <expression_t A, expression_t B>
  constexpr auto operator+(A&& a, B&& b) -> Tree<tree_size_v<A, B>>
  {
    return { new Sum(promote(FWD(a)), promote(FWD(b))) };
  }

  template <expression_t A>
  constexpr auto operator+(A&& a) -> decltype(auto)
  {
    return FWD(a);
  }

  template <expression_t A, expression_t B>
  constexpr auto operator-(A&& a, B&& b) -> Tree<tree_size_v<A, B>>
  {
    return { new Difference(promote(FWD(a)), promote(FWD(b))) };
  }

  template <expression_t A>
  constexpr auto operator-(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new Negate(promote(FWD(a))) };
  }

  template <expression_t A, expression_t B>
  constexpr auto operator*(A&& a, B&& b) -> Tree<tree_size_v<A, B>>
  {
    return { new Product(promote(FWD(a)), promote(FWD(b))) };
  }

  template <expression_t A, expression_t B>
  constexpr auto operator/(A&& a, B&& b) -> Tree<tree_size_v<A, B>>
  {
    return { new Ratio(promote(FWD(a)), promote(FWD(b))) };
  }

  template <expression_t A>
  constexpr auto D(A&& a, std::same_as<TensorIndex> auto... is) -> Tree<tree_size_v<A>>
  {
    return { new Partial(std::move(a), is...) };
  }

  constexpr auto δ(TensorIndex i, TensorIndex j) -> Tree<1>
  {
    return { new Delta(i + j) };
  }

  constexpr auto ε(std::same_as<TensorIndex> auto... is) -> Tree<1>
  {
    return { new Epsilon((is + ...)) };
  }

  template <expression_t A>
  constexpr auto abs(A&& a) -> Tree<tree_size_v<A>>
  {
    assert(a.rank() == 0);
    return { new CMath(promote(FWD(a)), ABS) };
  }

  template <expression_t A>
  constexpr auto fmin(A&& a, Rational q) -> Tree<tree_size_v<A>>
  {
    assert(a.rank() == 0);
    return { new CMath(promote(FWD(a)), std::move(q), FMIN) };
  }

  template <expression_t A>
  constexpr auto fmax(A&& a, Rational q) -> Tree<tree_size_v<A>>
  {
    assert(a.rank() == 0);
    return { new CMath(promote(FWD(a)), std::move(q), FMAX) };
  }

  template <expression_t A>
  constexpr auto exp(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), EXP) };
  }

  template <expression_t A>
  constexpr auto log(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), LOG) };
  }

  template <expression_t A>
  constexpr auto pow(A&& a, Rational q) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), std::move(q), LOG) };
  }

  template <expression_t A>
  constexpr auto sqrt(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), SQRT) };
  }

  template <expression_t A>
  constexpr auto sin(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), SIN) };
  }

  template <expression_t A>
  constexpr auto cos(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), COS) };
  }

  template <expression_t A>
  constexpr auto tan(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), TAN) };
  }

  template <expression_t A>
  constexpr auto asin(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), ASIN) };
  }

  template <expression_t A>
  constexpr auto acos(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), ACOS) };
  }

  template <expression_t A>
  constexpr auto atan(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), ATAN) };
  }

  template <expression_t A>
  constexpr auto atan2(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), ATAN2) };
  }

  template <expression_t A>
  constexpr auto sinh(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), SINH) };
  }

  template <expression_t A>
  constexpr auto cosh(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), COSH) };
  }

  template <expression_t A>
  constexpr auto tanh(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), TANH) };
  }

  template <expression_t A>
  constexpr auto asinh(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), ASINH) };
  }

  template <expression_t A>
  constexpr auto acosh(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), ACOSH) };
  }

  template <expression_t A>
  constexpr auto atanh(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), ATANH) };
  }

  template <expression_t A>
  constexpr auto ceil(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), CEIL) };
  }

  template <expression_t A>
  constexpr auto floor(A&& a) -> Tree<tree_size_v<A>>
  {
    return { new CMath(promote(FWD(a)), FLOOR) };
  }
}
