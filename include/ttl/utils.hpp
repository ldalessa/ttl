#pragma once

#include <ce/dvector.hpp>
#include <concepts>
#include <optional>
#include <utility>

namespace ttl::utils {
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

template <typename Range, typename T>
constexpr bool contains(Range&& range, T&& value) {
  return index_of(std::forward<Range>(range), std::forward<T>(value)).has_value();
}

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

template <typename T>
struct stack : ce::dvector<T> {
  using ce::dvector<T>::dvector;

  constexpr void  push(const T& t) { this->push_back(t); }
  constexpr void  push(T&& t) { this->push_back(std::move(t)); }
  constexpr    T  pop() { return this->pop_back(); }
  constexpr    T& top() { return this->back(); }
};

template <typename T>
struct set : ce::dvector<T> {
  using ce::dvector<T>::dvector;

  template <typename... Ts>
  constexpr void emplace(Ts&&... ts) {
    T& back = this->emplace_back(std::forward<Ts>(ts)...);
    if (index_of(*this, back) != size(*this) - 1) {
      this->pop_back();
    }
  }
};
}
