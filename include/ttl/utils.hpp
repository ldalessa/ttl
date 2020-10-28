#pragma once

#include <optional>
#include <utility>

namespace ttl::utils {
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

/// Simple wrapper for finding the index of a value in a range.
template <typename Range, typename T>
constexpr std::optional<int> index_of(Range&& range, T&& value) {
  auto begin = std::begin(range);
  auto   end = std::end(std::forward<Range>(range));
  auto     i = std::find(begin, end, std::forward<T>(value));
  if (i != end) {
    return std::distance(begin, i);
  }
  return std::nullopt;
}

// wrap the std integer sequence in something a little less verbose
template <auto... Ns>
struct seq : std::integer_sequence<std::common_type_t<decltype(Ns)...>, Ns...> {
  using std::integer_sequence<std::common_type_t<decltype(Ns)...>, Ns...>::integer_sequence;
  constexpr seq(std::integer_sequence<std::common_type_t<decltype(Ns)...>, Ns...>) {}
};

// create a sequence for [0, N).
template <auto N>
constexpr inline seq make_seq_v = std::make_integer_sequence<decltype(N), N>();

// integral constant is just a single element sequence
template <auto N>
constexpr inline seq<N> ic = {};

// basic debugging utility to print the types of expressions.
template <typename... Ts>
struct print_types_t;

template <typename... Ts>
void print_types(Ts...) { print_types_t<Ts...> _; }

}
