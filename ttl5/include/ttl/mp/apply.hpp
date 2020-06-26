#pragma once

#include <utility>

namespace ttl::mp {
/// Simple std::apply-like operation for sequences.
///
/// This supports a model where integer sequences look like tuples of
/// `integral_constant`s. Because the values are encoded as integral constants
/// they can be used in `constexpr` context inside of the `op`.
template <typename Op, typename T, T... Vs>
constexpr auto apply(Op&& op, std::integer_sequence<T, Vs...>) {
  return std::forward<Op>(op)(std::integral_constant<T, Vs>()...);
}

/// Convenience wrapper for sequence apply.
///
/// The supplied `N` will be expanded into an integer sequence and forwarded to
/// the basic `apply` operation.
template <auto N, typename Op>
constexpr auto apply(Op&& op) {
  return apply(std::forward<Op>(op), std::make_integer_sequence<decltype(N), N>());
}
}
