#pragma once

#include <cassert>
#include <concepts>
#include <optional>
#include <string_view>
#include <fmt/core.h>

namespace ttl {
  struct Index
  {
    char data[8] = {};
    int n = 0;

    constexpr Index() = default;

    constexpr Index(char c)
    {
      data[n++] = c;
    }

    constexpr Index(std::same_as<Index> auto const&... is)
    {
      ([&] {
        for (char c : is) {
          data[n++] = c;
        }
      }(), ...);
    }

    constexpr auto size() const -> int
    {
      return n;
    }

    constexpr auto begin() const { return std::begin(data); }
    constexpr auto begin()       { return std::begin(data); }
    constexpr auto   end() const { return begin() + n; }
    constexpr auto   end()       { return begin() + n; }

    constexpr const char& operator[](int i) const { return data[i]; }
    constexpr       char& operator[](int i)       { return data[i]; }

    constexpr void push_back(char c) {
      data[n++] = c;
    }

    constexpr friend bool operator==(Index const&, Index const&) = default;
    constexpr friend auto operator<=>(Index const&, Index const&) = default;

    // Count the number of `c` in the index.
    constexpr auto count(char c) const -> int
    {
      int cnt = 0;
      for (char d : data) {
        cnt += (c == d);
      }
      return cnt;
    }

    // Return the index of the first instance of `c` in the index, or nullopt.
    constexpr auto index_of(char c) const -> std::optional<int>
    {
      for (int i = 0, e = n; i < e; ++i) {
        if (c == data[i]) {
          return i;
        }
      }
      return std::nullopt;
    }

    // Hopefully obviously, search for chars in `search` and replace with the
    // corresponding char in `replace`.
    constexpr auto search_and_replace(Index const& search, Index const& replace)
      -> Index&
    {
      assert(search.size() == replace.size());
      for (char& c : data) {
        if (auto&& i = search.index_of(c)) {
          c = replace[*i];
        }
      }
      return *this;
    }
  };

  constexpr auto reverse(Index const& a) -> Index
  {
    Index out;
    for (int i = a.n - 1; i >= 0; --i) {
      out.push_back(a[i]);
    }
    return out;
  }

  constexpr auto unique(Index const& a) -> Index
  {
    Index out;
    for (char c : a) {
      if (out.count(c) == 0) {
        out.push_back(c);
      }
    }
    return out;
  }

  constexpr auto repeated(Index const& a) -> Index
  {
    Index out;
    for (char c : a) {
      if (a.count(c) > 1 && !out.index_of(c)) {
        out.push_back(c);
      }
    }
    return out;
  }

  constexpr auto exclusive(Index const& a) -> Index
  {
    Index out;
    for (char c : a) {
      if (a.count(c) == 1) {
        out.push_back(c);
      }
    }
    return out;
  }

  constexpr auto operator+=(Index& a, Index const& b) -> Index&
  {
    for (char c : b) a.push_back(c);
    return a;
  }

  constexpr auto operator+(Index const& a, Index const& b) -> Index
  {
    return { a, b };
  }

  constexpr auto operator&(Index const& a, Index const& b) -> Index
  {
    Index out;
    for (char c : a) {
      if (b.index_of(c)) {
        out.push_back(c);
      }
    }
    return out;
  }

  constexpr auto operator-(Index const& a, Index const& b) -> Index
  {
    Index out;
    for (char c : a) {
      if (!b.index_of(c)) {
        out.push_back(c);
      }
    }
    return out;
  }

  constexpr auto operator^(Index const& a, Index const& b) -> Index
  {
    return (a - b) + (b - a);
  }

  constexpr bool permutation(Index const& a, Index const& b) {
    return (a - b).n == 0 && (b - a).n == 0;
  }

  constexpr auto to_string(Index const& index) -> std::string_view
  {
    return { index.begin(), index.end() };
  }
}

template <>
struct fmt::formatter<ttl::Index>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  constexpr auto format(ttl::Index const& index, auto& ctx)
  {
    return format_to(ctx.out(), "{}", to_string(index));
  }
};
