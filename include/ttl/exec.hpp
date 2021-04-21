#pragma once

#include "ttl/ScalarIndex.hpp"
#include "ttl/pow.hpp"

#include <algorithm>
#include <array>
#include <cassert>

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

    constexpr friend bool operator==(Index const& a, Index const& b)
    {
      return (a.size() == b.size()) && std::equal(a.i, a.e, b.i, b.e);
    }

    constexpr friend auto operator<=>(Index const& a, Index const& b)
    {
      return std::lexicographical_compare_three_way(a.i, a.e, b.i, b.e);
    }

    constexpr auto operator[](int index) const -> char
    {
      return i[index];
    }

    constexpr auto size() const -> int
    {
      return e - i;
    }

    constexpr auto index_of(char c) const -> int
    {
      for (auto ii = i; ii < e; ++ii) {
        if (*ii == c) {
          return ii - i;
        }
      }
      assert(false);
    }
  };

  template <exec::Index const& from, exec::Index const& to, int N>
  struct IndexMapper
  {
    constexpr static int S = from.size();
    constexpr static int T = to.size();

    constexpr static std::array<int, T> selection_map_ = []
    {
      std::array<int, T> map;
      for (int i = 0; i < T; ++i) {
        map[i] = from.index_of(to[i]);
      }
      return map;
    }();

    constexpr static auto row_major(std::array<int, S> const& in)
    {
      int sum = 0;
      int   n = 1;
      for (int i = 0; i < T; ++i) {
        sum += n * in[selection_map_[i]];
        n *= N;
      }
      return sum;
    }

    /// Row-major expansion of the mapped result.
    constexpr static auto map(ScalarIndex const& index) -> int
    {
      assert(index.size() == S);
      std::array<int, S> in;
      for (int i = 0; i < S; ++i) {
        in[i] = index[i];
      }
      return row_major(in);
    }

    constexpr static auto make_map()
    {
      std::array<int, ttl::pow(N, S)> out;
      ScalarIndex i(S);
      int j = 0;
      do {
        out[j++] = map(i);
      } while (i.carry_sum_inc(N));
      return out;
    }

    constexpr static auto map_ = make_map();

    constexpr auto operator[](int i) const -> int
    {
      return map_[i];
    }

    constexpr static auto size() -> int
    {
      return std::ssize(map_);
    }
  };
} // namespace exec
