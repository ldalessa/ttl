#pragma once

#include "tensor.hpp"
#include <concepts>

namespace ttl
{
template <typename T> concept      Named = requires(T t) { name(t); };
template <typename T> concept    Ordered = requires(T t) { order(t); };
template <typename T> concept      Bound = requires(T t) { outer(t); };
template <typename T> concept     Parent = requires(T t) { children(t); };
template <typename T> concept Rewritable = true;
template <typename T> concept     Tensor = std::is_same_v<std::remove_cvref_t<T>, tensor>;
template <typename T> concept Expression = Named<T> && Ordered<T> && Bound<T> && Rewritable<T>;
template <typename T> concept   Internal = Expression<T> && Parent<T>;
template <typename T> concept       Node = Tensor<T> || Expression<T>;
}
