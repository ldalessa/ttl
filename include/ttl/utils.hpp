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
  return std::find(std::begin(range), std::end(range), value) != std::end(range);
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
constexpr static bool carry_sum_inc(int N, int n, int Order, Index&& index) {
  for (; n < Order; index[n++] = 0) {
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
  } while (utils::carry_sum_inc(N, 0, Order, index));
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
  constexpr std::optional<int> find(Ts&&... ts) {
    T temp(std::forward<Ts>(ts)...);
    auto i = std::lower_bound(this->begin(), this->end(), temp);
    if (i == this->end()) return std::nullopt;
    if (*i != temp) return std::nullopt;
    return i - this->begin();
  }

  template <typename... Ts>
  constexpr void emplace(Ts&&... ts) {
    T temp(std::forward<Ts>(ts)...);
    if (!contains(*this, temp)) {
      this->push_back(std::move(temp));
    }
//     // gcc produces different results in constexpr context for this code, I'm
//     // not sure why. I spent some time trying to reduce a testcase and couldn't
//     // get anything to fail.
//     //
//     const T& back = this->emplace_back(std::forward<Ts>(ts)...);
//     for (int i = 0; i < this->size() - 1; ++i) {
//       if (back == (*this)[i]) {
//         this->pop_back();
//         return;
//       }
//     }
  }
};


template <typename T>
struct box
{
  T* ptr = nullptr;

  constexpr box() = default;
  constexpr box(const box&) = delete;
  constexpr box& operator=(const box&) = delete;

  constexpr ~box() {
    delete ptr;
  }

  constexpr box(T* ptr) : ptr(ptr) {
  }

  constexpr box(box&& rhs) : ptr(std::exchange(rhs.ptr, nullptr)) {
  }

  constexpr box& operator=(box&& rhs) {
    delete ptr;
    ptr = std::exchange(rhs.ptr, nullptr);
    return *this;
  }

  constexpr operator T*() const {
    return ptr;
  }

  constexpr operator T*() {
    return ptr;
  }
};
}
