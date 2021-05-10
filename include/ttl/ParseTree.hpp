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
  struct ParseTree
  {
    using parse_tree_tag = void;

    node_ptr root;

    constexpr ParseTree(Node* ptr)
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
      -> ParseTree<A + 1>
    {
      return { new Exponent(root) }; // copy root
    }

    constexpr auto operator()(std::same_as<TensorIndex> auto... is) const
      -> ParseTree<A + 1>
    {
      return { new Bind(root, (is + ...)) }; // copy root
    }
  };

  template <class T>
  concept parse_tree = requires {
    typename std::remove_cvref_t<T>::parse_tree_tag;
  };

  template <class T>
  concept parse_expression =
    parse_tree<T> ||
    std::same_as<std::remove_cvref_t<T>, Rational> ||
    std::integral<T> ||
    std::floating_point<T>;

  template <parse_expression... Exprs>
  constexpr inline int parse_tree_size_v = (parse_tree_size_v<Exprs> + ... + 1);

  template <parse_expression Expr>
  constexpr inline int parse_tree_size_v<Expr> = 1;

  template <int A>
  constexpr inline int parse_tree_size_v<ParseTree<A>> = A;

  template <parse_expression... Exprs>
  using parse_tree_t = ParseTree<parse_tree_size_v<Exprs...>>;

  constexpr auto promote(parse_tree auto&& a) -> node_ptr
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

  template <parse_expression A, parse_expression B>
  constexpr auto operator+(A&& a, B&& b) -> parse_tree_t<A, B>
  {
    return { new Sum(promote(FWD(a)), promote(FWD(b))) };
  }

  template <parse_expression A>
  constexpr auto operator+(A&& a) -> decltype(auto)
  {
    return FWD(a);
  }

  template <parse_expression A, parse_expression B>
  constexpr auto operator-(A&& a, B&& b) -> parse_tree_t<A, B>
  {
    return { new Difference(promote(FWD(a)), promote(FWD(b))) };
  }

  template <parse_expression A>
  constexpr auto operator-(A&& a) -> parse_tree_t<A>
  {
    return { new Negate(promote(FWD(a))) };
  }

  template <parse_expression A, parse_expression B>
  constexpr auto operator*(A&& a, B&& b) -> parse_tree_t<A, B>
  {
    return { new Product(promote(FWD(a)), promote(FWD(b))) };
  }

  template <parse_expression A, parse_expression B>
  constexpr auto operator/(A&& a, B&& b) -> parse_tree_t<A, B>
  {
    return { new Ratio(promote(FWD(a)), promote(FWD(b))) };
  }

  template <parse_expression A>
  constexpr auto D(A&& a, std::same_as<TensorIndex> auto... is) -> parse_tree_t<A>
  {
    return { new Partial(std::move(a), is...) };
  }

  constexpr auto δ(TensorIndex i, TensorIndex j) -> ParseTree<1>
  {
    return { new Delta(i + j) };
  }

  constexpr auto ε(std::same_as<TensorIndex> auto... is) -> ParseTree<1>
  {
    return { new Epsilon((is + ...)) };
  }

  template <parse_expression A>
  constexpr auto abs(A&& a) -> parse_tree_t<A>
  {
    assert(a.rank() == 0);
    return { new CMath(promote(FWD(a)), ABS) };
  }

  template <parse_expression A>
  constexpr auto fmin(A&& a, Rational q) -> parse_tree_t<A>
  {
    assert(a.rank() == 0);
    return { new CMath(promote(FWD(a)), std::move(q), FMIN) };
  }

  template <parse_expression A>
  constexpr auto fmax(A&& a, Rational q) -> parse_tree_t<A>
  {
    assert(a.rank() == 0);
    return { new CMath(promote(FWD(a)), std::move(q), FMAX) };
  }

  template <parse_expression A>
  constexpr auto exp(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), EXP) };
  }

  template <parse_expression A>
  constexpr auto log(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), LOG) };
  }

  template <parse_expression A>
  constexpr auto pow(A&& a, Rational q) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), std::move(q), LOG) };
  }

  template <parse_expression A>
  constexpr auto sqrt(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), SQRT) };
  }

  template <parse_expression A>
  constexpr auto sin(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), SIN) };
  }

  template <parse_expression A>
  constexpr auto cos(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), COS) };
  }

  template <parse_expression A>
  constexpr auto tan(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), TAN) };
  }

  template <parse_expression A>
  constexpr auto asin(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ASIN) };
  }

  template <parse_expression A>
  constexpr auto acos(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ACOS) };
  }

  template <parse_expression A>
  constexpr auto atan(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ATAN) };
  }

  template <parse_expression A>
  constexpr auto atan2(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ATAN2) };
  }

  template <parse_expression A>
  constexpr auto sinh(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), SINH) };
  }

  template <parse_expression A>
  constexpr auto cosh(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), COSH) };
  }

  template <parse_expression A>
  constexpr auto tanh(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), TANH) };
  }

  template <parse_expression A>
  constexpr auto asinh(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ASINH) };
  }

  template <parse_expression A>
  constexpr auto acosh(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ACOSH) };
  }

  template <parse_expression A>
  constexpr auto atanh(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), ATANH) };
  }

  template <parse_expression A>
  constexpr auto ceil(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), CEIL) };
  }

  template <parse_expression A>
  constexpr auto floor(A&& a) -> parse_tree_t<A>
  {
    return { new CMath(promote(FWD(a)), FLOOR) };
  }
}
