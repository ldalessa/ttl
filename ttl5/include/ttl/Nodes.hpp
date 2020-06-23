#pragma once

#include <utility>

namespace ttl {
enum NodeType : char {
  SUM = 's',
  PRODUCT = 'p',
  INVERSE = 'i',
  BIND = 'b',
  PARTIAL = 'x',
  DELTA = 'd',
  TENSOR = 't',
  RATIONAL = 'q',
  DOUBLE = 'd'
};

template <char N>
using Node = std::integral_constant<char, N>();

template <typename T>
constexpr inline char node_type_v = '\0';

template <char N>
constexpr inline char node_type_v<Node<N>> = N;

template <typename T>
concept Leaf =
 std::is_same_v<std::remove_cvref_t<T>, std::integral_constant<char, DELTA>> ||
 std::is_same_v<std::remove_cvref_t<T>, std::integral_constant<char, TENSOR>> ||
 std::is_same_v<std::remove_cvref_t<T>, std::integral_constant<char, RATIONAL>> ||
 std::is_same_v<std::remove_cvref_t<T>, std::integral_constant<char, DOUBLE>>;

template <typename T>
concept Unary =
 std::is_same_v<std::remove_cvref_t<T>, std::integral_constant<char, BIND>> ||
 std::is_same_v<std::remove_cvref_t<T>, std::integral_constant<char, PARTIAL>>;

template <typename T>
concept Binary =
 std::is_same_v<std::remove_cvref_t<T>, std::integral_constant<char, SUM>> ||
 std::is_same_v<std::remove_cvref_t<T>, std::integral_constant<char, PRODUCT>> ||
 std::is_same_v<std::remove_cvref_t<T>, std::integral_constant<char, INVERSE>>;
}

