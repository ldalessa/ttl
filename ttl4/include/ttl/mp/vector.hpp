#pragma once

#include <algorithm>

namespace ttl::mp {
template <typename T>
class vector {
  int i_ = 0;
  int n_;
  T* data_;

  constexpr void grow() {
    T* old = std::exchange(data_, new T[n_ *= 2]);
    std::copy_n(old, i_, data_);
    delete [] old;
  }

 public:
  constexpr vector()
    : n_(1)
    , data_(new T[1])
  {
  }

  constexpr vector(const vector& v)
      : i_(v.i_)
      , n_(v.n_)
      , data_(new T[n_])
  {
    std::copy_n(v.data_, i_, data_);
  }

  constexpr vector(vector&& v)
      : i_(std::exchange(v.i_, 0))
      , n_(std::exchange(v.n_, 0))
      , data_(std::exchange(v.data_, nullptr))
  {
  }

  constexpr ~vector() {
    delete [] data_;
  }

  constexpr explicit vector(int n)
      : n_(n)
      , data_(new T[n_])
  {
  }

  constexpr int size() const {
    return i_;
  }

  constexpr int capacity() const {
    return n_;
  }

  constexpr void push(const T& t) {
    if (i_ + 1 == n_) {
      grow();
    }
    data_[i_++] = t;
  }

  constexpr void push(T&& t) {
    if (i_ + 1 == n_) {
      grow();
    }
    data_[i_++] = std::move(t);
  }

  template <typename... Ts>
  constexpr void emplace(Ts&&... ts) {
    if (i_ + 1 == n_) {
      grow();
    }
    data_[i_++] = T { std::forward<Ts>(ts)... };
  }

  constexpr T pop() {
    assert(i_ > 0);
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
