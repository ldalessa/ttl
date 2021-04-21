#pragma once

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
    constexpr static std::size_t S = from.size();
    constexpr static std::size_t T = to.size();

    constexpr static std::array<int, T> map_ = []
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
        sum += n * in[map_[i]];
        n *= N;
      }
      return sum;
    }

    /// Row-major expansion of the mapped result.
    template <std::size_t... is>
    constexpr auto operator()(std::index_sequence<is...>) const -> int
    {
      static_assert(sizeof...(is) == S);
      constexpr std::array<int, S> in = { is... };
      constexpr int i = row_major(in);
      return i;
    }

    constexpr auto operator()(std::integral auto... is) const -> int
    {
      static_assert(sizeof...(is) == S);
      std::array<int, S> in = { is... };
      int i = row_major(in);
      return i;
    }
  };

  /// Evaluate an operator for the M^N outer product of indices.
#ifdef TTL_CONSTEXPR_MAP
  template <int N, int M, std::size_t... is>
  constexpr void eval(auto const& op)
  {
    constexpr int m = sizeof...(is);
    if constexpr (m == M) {
      op(std::index_sequence<is...>());
    }
    else {
      [&]<std::size_t... i>(std::index_sequence<i...>) {
        (eval<N, M, i, is...>(op), ...);
      }(std::make_index_sequence<N>());
    }
  }
#else
  template <int N, int M>
  constexpr void eval(auto const& op, auto... is)
  {
    constexpr int m = sizeof...(is);
    if constexpr (m == M) {
      op(is...);
    }
    else {
      for (int i = 0; i < N; ++i) {
        eval<N, M>(op, i, is...);
      }
    }
  }
#endif
} // namespace exec
