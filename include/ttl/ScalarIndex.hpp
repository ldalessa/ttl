#pragma once

#include "Index.hpp"
#include <fmt/core.h>
#include <array>
#include <concepts>

namespace ttl
{
  struct ScalarIndex
  {
    int size_ = 0;
    std::array<int, 9> data_ = {};              // using array allows default <=>

    constexpr ScalarIndex() = default;

    constexpr ScalarIndex(int n)
        : size_(n)
    {
      assert(0 <= n and n < data_.size());
    }

    constexpr ScalarIndex(std::in_place_t, std::integral auto... is)
        : size_{ sizeof...(is) }
        , data_{ int(is)... }
    {
      static_assert(sizeof...(is) < std::size(data_));
      assert(((0 <= is) && ...));
    }

    constexpr auto  size() const
    {
      return size_;
    }

    constexpr auto begin() const
    {
      return std::begin(data_);
    }

    constexpr auto end() const
    {
      return begin() + size();
    }

    constexpr auto operator[](int i) const -> decltype(auto)
    {
      return data_[i];
    }

    constexpr auto operator[](int i) -> decltype(auto)
    {
      return data_[i];
    }

    constexpr void resize(int n) {
      size_ = n;
    }

    constexpr auto select(Index const& from, Index const& to) const
      -> ScalarIndex
    {
      assert(size_ == from.size());
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

    constexpr friend bool operator==(ScalarIndex const&, ScalarIndex const&) = default;
    constexpr friend auto operator<=>(ScalarIndex const&, ScalarIndex const&) = default;

    constexpr bool carry_sum_inc(int N)
    {
      for (int i = 0; i < size_; ++i) {
        if (++data_[i] < N) {
          return true;                          // no carry
        }
        data_[i] = 0;                           // reset and carry
      }
      return false;                             // overflow
    }

    constexpr bool carry_sum_inc(int N, int n)
    {
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
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  constexpr auto format(ttl::ScalarIndex const& index, auto& ctx)
  {
    auto out = ctx.out();
    for (auto&& i : index) {
      out = format_to(out, "{}", i);
    }
    return out;
  }
};
