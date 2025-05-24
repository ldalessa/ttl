#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <format>
#include <optional>
#include <string_view>

namespace ttl
{
  struct Index
  {
    int  size_ = 0;
    char data_[TTL_MAX_PARSE_INDEX] = {};

    constexpr Index() = default;

    constexpr Index(char c)
    {
      data_[size_++] = c;
    }

    constexpr Index(std::same_as<Index> auto const&... is)
    {
      ([&] {
        for (char c : is) {
          data_[size_++] = c;
        }
      }(), ...);
    }

    constexpr friend bool operator==(Index const& a, Index const& b)
    {
      return (a.size_ == b.size_) && std::equal(
        std::begin(a.data_), std::begin(a.data_) + a.size_,
        std::begin(b.data_), std::begin(b.data_) + b.size_);
    }

    constexpr friend auto operator<=>(Index const& a, Index const& b)
    {
      return std::lexicographical_compare_three_way(
        std::begin(a.data_), std::begin(a.data_) + a.size_,
        std::begin(b.data_), std::begin(b.data_) + b.size_);
    }

    constexpr auto size() const -> int
    {
      return size_;
    }

    constexpr auto begin() const -> decltype(auto)
    {
      return std::begin(data_);
    }

    constexpr auto begin() -> decltype(auto)
    {
      return std::begin(data_);
    }

    constexpr auto end() const -> decltype(auto)
    {
      return begin() + size_;
    }

    constexpr auto end() -> decltype(auto)
    {
      return begin() + size_;
    }

    constexpr auto operator[](int i) const -> const char&
    {
      return data_[i];
    }

    constexpr auto operator[](int i) -> char&
    {
      return data_[i];
    }

    constexpr void push_back(char c)
    {
      data_[size_++] = c;
    }

    // Count the number of `c` in the index.
    constexpr auto count(char c) const -> int
    {
      int cnt = 0;
      for (char d : data_) {
        cnt += (c == d);
      }
      return cnt;
    }

    // Return the index of the first instance of `c` in the index, or nullopt.
    constexpr auto index_of(char c) const -> std::optional<int>
    {
      for (int i = 0, e = size_; i < e; ++i) {
        if (c == data_[i]) {
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
      for (char& c : data_) {
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
    for (int i = a.size_ - 1; i >= 0; --i) {
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
    return (a - b).size_ == 0 && (b - a).size_ == 0;
  }

  constexpr auto to_string(Index const& index) -> std::string_view
  {
    return { index.begin(), index.end() };
  }
}

template <>
struct std::formatter<ttl::Index>
{
  static constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  static constexpr auto format(ttl::Index const& index, auto& ctx)
  {
    return format_to(ctx.out(), "{}", to_string(index));
  }
};
