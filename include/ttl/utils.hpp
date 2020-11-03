#pragma once

#include <ce/cvector.hpp>
#include <ce/dvector.hpp>
#include <concepts>
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

// basic debugging utility to print the types of expressions.
template <typename... Ts>
struct print_types_t;

template <typename... Ts>
void print_types(Ts...) { print_types_t<Ts...> _; }

template <std::integral T>
constexpr T pow(T x, T y) {
  assert(y >= 0);
  T out = 1;
  for (T i = 0; i < y; ++i) {             // slow, buy hey, it's constexpr :-)
    out *= x;
  }
  return out;
}

template <typename Index>
constexpr static bool carry_sum_inc(int N, int Order, Index&& index) {
  for (int n = 0; n < Order; index[n++] = 0) {
    if (++index[n] < N) {
      return true; // no carry
    }
  }
  return false;
}

template <typename Op>
constexpr static void expand(int N, int Order, Op&& op) {
  int *index = new int[Order]{};
  do {
    op(index);
  } while (utils::carry_sum_inc(N, Order, index));
  delete [] index;
}

template <typename T, typename... Ts>
constexpr static std::array<T, sizeof...(Ts)> make_array(Ts&&... ts) {
  return { std::forward<Ts>(ts)... };
}

template <typename T, int N>
struct set : ce::cvector<T, N> {
  using ce::cvector<T, N>::cvector;

  template <typename... Ts>
  constexpr void emplace(Ts&&... ts) {
    auto& back = this->emplace_back(std::forward<Ts>(ts)...);
    if (index_of(*this, back) != size(*this) - 1) {
      this->pop_back();
    }
  }

  constexpr set& sort() {
    std::sort(this->begin(), this->end());
    return *this;
  }
};

// template <typename T, int N>
// struct stack : ce::cvector<T, N> {
//   using ce::cvector<T, N>::cvector;

//   constexpr void push(const T& t) { this->push_back(t); }
//   constexpr void push(T&& t) { this->push_back(std::move(t)); }
//   constexpr    T pop() { return this->pop_back(); }
// };

template <typename T>
struct stack : ce::dvector<T> {
  using ce::dvector<T>::dvector;

  constexpr void push(const T& t) { this->push_back(t); }
  constexpr void push(T&& t) { this->push_back(std::move(t)); }
  constexpr    T pop() { return this->pop_back(); }
};
}
