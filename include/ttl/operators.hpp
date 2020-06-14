#pragma once

#include "expression.hpp"
#include "nodes.hpp"

namespace ttl
{
template <typename T>
concept Constant = std::integral<T> || std::floating_point<T> || Rational<T> || Delta<T>;

template <Expression A, Zero B>
constexpr auto operator+(A a, B) {
  return a;
}

template <Zero A, Expression B>
constexpr auto operator+(A, B b) {
  return b;
}

template <Expression A, Expression B>
constexpr auto operator+(A a, B b) {
  return sum(a, b);
}

template <Expression A>
constexpr auto operator+(A a, const tensor& b) {
  return a + bind(b);
}

template <Expression B>
constexpr auto operator+(const tensor& a, B b) {
  return bind(a) + b;
}

constexpr auto operator+(const tensor& a, const tensor& b) {
  return bind(a) + bind(b);
}

template <Expression A, One B>
constexpr auto operator*(A a, B) {
  return a;
}

template <One A, Expression B>
constexpr auto operator*(A, B b) {
  return b;
}

template <Expression A, Zero B>
constexpr auto operator*(A, B b) {
  return b;
}

template <Zero A, Expression B>
constexpr auto operator*(A a, B) {
  return a;
}

template <Expression A, Expression B>
constexpr auto operator*(A a, B b) {
  return product(a, b);
}

template <Expression A>
constexpr auto operator*(A a, const tensor& b) {
  return a * bind(b);
}

template <Expression B>
constexpr auto operator*(const tensor& a, B b) {
  return bind(a) * b;
}

constexpr auto operator*(const tensor& a, const tensor& b) {
  return bind(a) * bind(b);
}

template <Node A>
constexpr auto operator-(A&& a) {
  return neg_one * std::forward<A>(a);
}

template <Node A, Node B>
constexpr auto operator-(A&& a, B&& b) {
  return std::forward<A>(a) + (-std::forward<B>(b));
}

template <Node A, Expression B>
constexpr auto operator/(A&& a, B b) {
  return std::forward<A>(a) * inverse(b);
}

template <Node A>
constexpr auto operator/(A&& a, const tensor& b) {
  return std::forward<A>(a) / bind(b);
}

template <Node A, Arithmetic B>
constexpr auto operator/(A&& a, B b) {
  assert(b != 0);
  return std::forward<A>(a) * (B{1} / b);
}

template <Node A, Rational B>
constexpr auto operator/(A&& a, B b) {
  constexpr rational<1, 1> one;
  return std::forward<A>(a) * (one / b);
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
    auto&& [l, r] = children(a);
    return D(l, is...) + D(r, is...);
  }
  else if constexpr (Product<A>) {
    auto&& [l, r] = children(a);
    return D(l, is...) * r + l * D(r, is...);
  }
  else if constexpr (Partial<A>) {
    return a.append(is...);
  }
  else {
    return partial(std::forward<A>(a), (is + ...));
  }
}
}
