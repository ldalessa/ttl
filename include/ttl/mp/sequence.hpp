#pragma once

#include <concepts>
#include <utility>

namespace ttl::mp {
template <std::signed_integral T, T V>
struct cv {
  constexpr cv() = default;
  constexpr cv(std::integral_constant<T, V>) {}
  constexpr operator T() const { return V; }
};

template <auto N>
inline constexpr cv cv_v = cv<decltype(N), N>();

template <std::signed_integral T, T... Vs>
struct cs {
  constexpr cs() = default;
  constexpr cs(std::integer_sequence<T, Vs...>) {}

  static constexpr T data[] = { Vs... };

  constexpr auto begin() const { return data; }
  constexpr auto   end() const { return data + sizeof...(Vs); }
};

template <auto N>
inline constexpr cs cs_v = cs(std::make_integer_sequence<decltype(N), N>());

/// Simple std::apply-like operation for sequences.
///
/// This supports a model where integer sequences look like tuples of
/// `integral_constant`s. Because the values are encoded as integral constants
/// they can be used in `constexpr` context inside of the `op`.
template <typename Op, std::signed_integral T, T... Vs>
constexpr auto apply(Op&& op, cs<T, Vs...>) {
  return std::forward<Op>(op)(cv_v<Vs>...);
}

/// Convenience wrapper for sequence apply.
///
/// The supplied `N` will be expanded into an integer sequence and forwarded to
/// the basic `apply` operation.
template <auto N, typename Op>
constexpr auto apply(Op&& op) {
  return apply(std::forward<Op>(op), cs_v<N>);
}
}
