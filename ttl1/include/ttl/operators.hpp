#pragma once

#include "expression.hpp"
#include "nodes.hpp"

namespace ttl
{
template <typename T>
concept Constant = Arithmetic<T> || Rational<T> || Delta<T>;

template <Node A, Node B>
constexpr auto operator+(A&& a, B&& b) {
  if constexpr (is_zero_v<A>) {
    return b;
  }
  else if constexpr (is_zero_v<B>) {
    return a;
  }
  else if constexpr (Tensor<B>) {
    return std::forward<A>(a) + bind(std::forward<B>(b));
  }
  else if constexpr (Tensor<A>) {
    return bind(std::forward<A>(a)) + std::forward<B>(b);
  }
  else {
    return sum(std::forward<A>(a), std::forward<B>(b));
  }
}

template <Node A, Node B>
constexpr auto operator*(A&& a, B&& b) {
  if constexpr (Zero<B>) {
    return b;
  }
  else if constexpr (Zero<A>) {
    return a;
  }
  if constexpr (One<B>) {
    return a;
  }
  else if constexpr (One<A>) {
    return b;
  }
  else if constexpr (Tensor<B>) {
    return std::forward<A>(a) * bind(std::forward<B>(b));
  }
  else if constexpr (Tensor<A>) {
    return bind(std::forward<A>(a)) * std::forward<B>(b);
  }
  else {
    return product(std::forward<A>(a), std::forward<B>(b));
  }
}

template <Node A, Node B>
constexpr auto operator/(A&& a, B&& b) {
  static_assert(!Zero<B>);
  if constexpr (Zero<A> || One<B>) {
    return a;
  }
  else if constexpr (Rational<B>) {
    return std::forward<A>(a) * (one / b);
  }
  else if constexpr (Arithmetic<B>) {
    return std::forward<A>(a) * (1 / b);
  }
  else if constexpr (Tensor<B>) {
    return std::forward<A>(a) / bind(std::forward<B>(b));
  }
  else if constexpr (Inverse<B>) {
    // @note assuming b has an inverse
    return std::apply([&](auto&& c) {
      return std::forward<A>(a)> * std::forward<decltype(c)>(c);
    }, children(b));
  }
  else {
    return std::forward<A>(a) * inverse(b);
  }
}

template <Node A, Index... Is>
constexpr auto D(A&& a, Is... is) {
  if constexpr (Constant<A>) {
    return zero;
  }
  else if constexpr (Tensor<A>) {
    return D(bind(std::forward<A>(a)), is...);
  }
  else if constexpr (Sum<A>) {
    auto&& [l, r] = children(std::forward<A>(a));
    using L = decltype(l);
    using R = decltype(r);
    return D(std::forward<L>(l), is...) + D(std::forward<R>(r), is...);

  }
  else if constexpr (Product<A>) {
    // Product rule, but don't produce extra stuff if one of the arms is going
    // to be zero. This is both inefficient and causes us problems because the
    // `operator+()` is going to have trouble because the arm with 0 is going to
    // have a different order.
    auto&& [l, r] = children(std::forward<A>(a));
    using L = decltype(l);
    using R = decltype(r);
    if constexpr (Constant<L>) {
      return std::forward<L>(l) * D(std::forward<R>(r), is...);
    }
    else if constexpr (Constant<R>) {
      return D(std::forward<L>(l), is...) * std::forward<R>(r);
    }
    else {
      return D(std::forward<L>(l), is...) * std::forward<R>(r) + std::forward<L>(l) * D(std::forward<R>(r), is...);
    }
  }
  else if constexpr (Partial<A>) {
    return std::forward<A>(a).append(is...);
  }
  else {
    return partial(std::forward<A>(a), (is + ...));
  }
}

template <Node A>
constexpr auto operator+(A&& a) {
  return a;
}

template <Node A>
constexpr auto operator-(A&& a) {
  return neg_one * std::forward<A>(a);
}

template <Node A, Node B>
constexpr auto operator-(A&& a, B&& b) {
  return std::forward<A>(a) + (-std::forward<B>(b));
}
}
