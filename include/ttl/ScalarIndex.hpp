#pragma once

#include "Index.hpp"
#include <fmt/core.h>
#include <concepts>

namespace ttl {
struct ScalarIndex
{
 private:
  int size_ = 0;
  int data_[9] = {};

 public:
  constexpr ScalarIndex() = default;
  constexpr ScalarIndex(int n) : size_(n) {}
  constexpr ScalarIndex(std::in_place_t, std::integral auto... is)
      : size_{ sizeof...(is) }
      , data_{ int(is)... }
  {
    assert(((0 <= is && is < std::numeric_limits<int>::max()) && ...));
  }

  constexpr auto  size() const { return size_; }
  constexpr auto begin() const { return data_; }
  constexpr auto   end() const { return data_ + size_; }

  constexpr const int& operator[](int i) const {
    return data_[i];
  }

  constexpr int& operator[](int i) {
    return data_[i];
  }

  constexpr void resize(int n) {
    size_ = n;
  }

  constexpr ScalarIndex
  select(const Index& from, const Index& to) const
  {
    assert(this->size_ == from.size());
    ScalarIndex out(to.size());
    for (int i = 0, e = to.size(); i < e; ++i) {
      for (int j = 0, e = from.size(); j < e; ++j) {
        if (to[i] == from[j]) {
          out[i] = data_[j];
        }
      }
    }
    return out;
  }

  constexpr friend bool operator==(const ScalarIndex& a, const ScalarIndex& b) {
    if (a.size() != b.size()) return false;
    for (int i = 0, e = a.size(); i < e; ++i) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  constexpr friend bool operator<(const ScalarIndex& a, const ScalarIndex& b) {
    if (a.size() < b.size()) return true;
    if (b.size() < a.size()) return false;
    for (int i = 0, e = a.size(); i < e; ++i) {
      if (a[i] < b[i]) return true;
      if (b[i] < a[i]) return false;
    }
    return false;
  }

  constexpr bool carry_sum_inc(int N, int n) {
    for (; n < size_; data_[n++] = 0) {
      if (++data_[n] < N) {
        return true; // no carry
      }
    }
    return false;
  }
};
}

template <>
struct fmt::formatter<ttl::ScalarIndex>
{
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::ScalarIndex& index, FormatContext& ctx)
  {
    auto out = ctx.out();
    for (auto&& i : index) {
      out = format_to(out, "{}", i);
    }
    return out;
  }
};
