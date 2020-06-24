#pragma once

#include <cassert>
#include <utility>

namespace ttl::mp {
template <typename T, int M>
class cvector {
  int i_ = 0;
  T data_[M] = {};

 public:
  constexpr int size() const {
    return i_;
  }

  constexpr int capacity() const {
    return M;
  }

  constexpr void push(const T& t) {
    assert(0 <= i_ && i_ < M);
    data_[i_++] = t;
  }

  constexpr void push(T&& t) {
    assert(0 <= i_ && i_ < M);
    data_[i_++] = std::move(t);
  }

  template <typename... Ts>
  constexpr void emplace(Ts&&... ts) {
    assert(0 <= i_ && i_ < M);
    data_[i_++] = T { std::forward<Ts>(ts)... };
  }

  constexpr T pop() {
    assert(0 < i_);
    return data_[--i_];
  }

  constexpr const T& back() const {
    assert(i_ > 0);
    return data_[i_ - 1];
  }

  constexpr T& back() {
    assert(i_ > 0);
    return data_[i_ - 1];
  }
};
}
