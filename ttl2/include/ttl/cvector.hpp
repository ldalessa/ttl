#pragma once

#include <cassert>
#include <utility>

namespace ttl {
template <typename T, int M>
class cvector {
  T data[M] = {};
  int n = 0;

 public:
  constexpr cvector() = default;

  constexpr int size() const {
    return n;
  }

  constexpr void push(T t) {
    assert(0 <= n && n < M);
    data[n++] = t;
  }

  template <typename... Ts>
  constexpr void emplace(Ts&&... ts) {
    assert(0 <= n && n < M);
    data[n++] = T{std::forward<Ts>(ts)...};
  }

  constexpr T pop() {
    assert(0 < n);
    return data[--n];
  }
};
}
