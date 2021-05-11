#pragma once

#include "ttl/Nodes.hpp"
#include "ttl/Rational.hpp"
#include "ttl/TensorIndex.hpp"
#include <concepts>

#ifndef FWD
#define FWD(a) std::forward<decltype(a)>(a)
#endif

namespace ttl::parse
{
  template <int A>
  struct Tree
  {
    using tree_tag = void;

    node_ptr root;

    constexpr Tree(Node* ptr)
        : root(ptr)
    {
      assert(A == ptr->size());
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
  concept tree = requires {
    typename std::remove_cvref_t<T>::tree_tag;
  };

  template <class T>
  concept expression =
    tree<T> ||
    std::same_as<std::remove_cvref_t<T>, Rational> ||
    std::integral<T> ||
    std::floating_point<T>;

  template <expression... Exprs>
  constexpr inline int tree_size_v = (tree_size_v<Exprs> + ... + 1);

  template <expression Expr>
  constexpr inline int tree_size_v<Expr> = 1;

  template <int A>
  constexpr inline int tree_size_v<Tree<A>> = A;

  template <expression... Exprs>
  using tree_t = Tree<(tree_size_v<std::remove_cvref_t<Exprs>> + ... + 1)>;

  constexpr auto promote(tree auto&& a) -> node_ptr
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

  template <expression A, expression B>
  constexpr auto operator+(A&& a, B&& b) -> tree_t<A, B>
  {
    return { new Sum(promote(FWD(a)), promote(FWD(b))) };
  }

  template <expression A>
  constexpr auto operator+(A&& a) -> decltype(auto)
  {
    return FWD(a);
  }

  template <expression A, expression B>
  constexpr auto operator-(A&& a, B&& b) -> tree_t<A, B>
  {
    return { new Difference(promote(FWD(a)), promote(FWD(b))) };
  }

  template <expression A>
  constexpr auto operator-(A&& a) -> tree_t<A>
  {
    return { new Negate(promote(FWD(a))) };
  }

  template <expression A, expression B>
  constexpr auto operator*(A&& a, B&& b) -> tree_t<A, B>
  {
    return { new Product(promote(FWD(a)), promote(FWD(b))) };
  }

  template <expression A, expression B>
  constexpr auto operator/(A&& a, B&& b) -> tree_t<A, B>
  {
    return { new Ratio(promote(FWD(a)), promote(FWD(b))) };
  }

  template <expression A>
  constexpr auto D(A&& a, std::same_as<TensorIndex> auto... is) -> tree_t<A>
  {
    return { new Partial(std::move(a), is...) };
  }

  template <expression A>
  constexpr auto abs(A&& a) -> tree_t<A>
  {
    assert(a.rank() == 0);
    return { new CMath(promote(FWD(a)), ABS) };
  }

  template <expression A>
  constexpr auto fmin(A&& a, Rational q) -> tree_t<A>
  {
    assert(a.rank() == 0);
    return { new CMath(promote(FWD(a)), std::move(q), FMIN) };
  }

  template <expression A>
  constexpr auto fmax(A&& a, Rational q) -> tree_t<A>
  {
    assert(a.rank() == 0);
    return { new CMath(promote(FWD(a)), std::move(q), FMAX) };
  }

  template <expression A>
  constexpr auto exp(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), EXP) };
  }

  template <expression A>
  constexpr auto log(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), LOG) };
  }

  template <expression A>
  constexpr auto pow(A&& a, Rational q) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), std::move(q), LOG) };
  }

  template <expression A>
  constexpr auto sqrt(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), SQRT) };
  }

  template <expression A>
  constexpr auto sin(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), SIN) };
  }

  template <expression A>
  constexpr auto cos(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), COS) };
  }

  template <expression A>
  constexpr auto tan(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), TAN) };
  }

  template <expression A>
  constexpr auto asin(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ASIN) };
  }

  template <expression A>
  constexpr auto acos(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ACOS) };
  }

  template <expression A>
  constexpr auto atan(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ATAN) };
  }

  template <expression A>
  constexpr auto atan2(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ATAN2) };
  }

  template <expression A>
  constexpr auto sinh(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), SINH) };
  }

  template <expression A>
  constexpr auto cosh(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), COSH) };
  }

  template <expression A>
  constexpr auto tanh(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), TANH) };
  }

  template <expression A>
  constexpr auto asinh(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ASINH) };
  }

  template <expression A>
  constexpr auto acosh(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ACOSH) };
  }

  template <expression A>
  constexpr auto atanh(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ATANH) };
  }

  template <expression A>
  constexpr auto ceil(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), CEIL) };
  }

  template <expression A>
  constexpr auto floor(A&& a) -> tree_t<A>
  {
    return { new CMath(promote(FWD(a)), FLOOR) };
  }
}

namespace ttl
{
  constexpr auto δ(TensorIndex i, TensorIndex j) -> parse::Tree<1>
  {
    return { new parse::Delta(i + j) };
  }

  constexpr auto ε(std::same_as<TensorIndex> auto... is) -> parse::Tree<1>
  {
    return { new parse::Epsilon((is + ...)) };
  }
}
