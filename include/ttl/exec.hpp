#pragma once

#include <array>

namespace ttl::exec
{
  enum Tag : int {
    SUM,
    DIFFERENCE,
    PRODUCT,
    RATIO,
    IMMEDIATE,
    SCALAR,
    CONSTANT,
    DELTA
    };

  constexpr bool is_binary(Tag tag)
  {
    return tag < IMMEDIATE;
  }

  struct Index
  {
    const char* i;
    const char* e;

    constexpr auto size() const -> int
    {
      return e - i;
    }

    constexpr char operator[](int index) const
    {
      return i[index];
    }

    constexpr friend bool operator==(Index const& a, Index const& b)
    {
      if (a.size() != b.size()) {
        return false;
      }

      for (int j = 0, e = a.size(); j < e; ++j)
      {
        if (a.i[j] != b.i[j]) {
          return false;
        }
      }

      return true;
    }
  };

  template <auto const& from, auto const& to>
  struct IndexMapper
  {
    constexpr auto operator()(std::array<int, from.size()> index) const
      -> std::array<int, to.size()>
    {
      if constexpr (from == to) {
          return index;
        }
      else {
        std::array<int, to.size()> out;
        for (int i = 0; i < to.size(); ++i) {
          out[i] = 0;
        }
        return out;
      }
    }
  };

  template <int N, std::size_t M>
  constexpr int row_major(std::array<int, M> const& index)
  {
    int sum = 0;
    int n = 1;
    for (int i : index) {
      if (i < 0 or N <= i) __builtin_unreachable();
      sum += n * i;
      n *= N;
    }
    return sum;
  }

  template <int N, std::size_t M, int m = 0>
  constexpr void eval(auto const& op, std::array<int, M> index = {}) {
    if constexpr (m == M) {
        op(index);
      }
    else {
      for (int i = 0; i < N; ++i) {
        index[m] = i;
        eval<N, M, m+1>(op, index);
      }
    }
  }
} // namespace exec
