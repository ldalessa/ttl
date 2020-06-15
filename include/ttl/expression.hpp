#pragma once

#include <tuple>

namespace ttl
{
template <typename T, auto N>
concept Cardinality = std::tuple_size_v<std::remove_cvref_t<T>> == N;

template <typename T>
concept UnaryNode = requires(T t) {
  { children(t) } -> Cardinality<1>;
};

template <typename T>
concept BinaryNode = requires(T t) {
  { children(t) } -> Cardinality<2>;
};

template <typename T> concept      Named = requires(T t) { name(t); };
template <typename T> concept    Ordered = requires(T t) { order(t); };
template <typename T> concept      Bound = requires(T t) { outer(t); };
template <typename T> concept     Tensor = Named<T> && Ordered<T> && not Bound<T>;
template <typename T> concept Expression = Named<T> && Ordered<T>;
template <typename T> concept   Internal = Expression<T> && (UnaryNode<T> || BinaryNode<T>);
template <typename T> concept       Node = Tensor<T> || Expression<T>;
}
