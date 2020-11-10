#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <optional>
#include <span>
#include <string_view>
#include <fmt/core.h>

namespace ttl {
struct Index
{
  char data[8] = {};
  int n = 0;

  constexpr Index() = default;
  constexpr Index(char c) {
    data[n++] = c;
  }

  constexpr Index(std::same_as<Index> auto const&... is) {
    ([&] { for (auto c : is) data[n++] = c; }(), ...);
  }

  constexpr int size() const { return n; }

  constexpr const char* begin() const { return data; }
  constexpr const char*   end() const { return data + n; }
  constexpr       char* begin()       { return data; }
  constexpr       char*   end()       { return data + n; }

  constexpr const char& operator[](int i) const { return data[i]; }
  constexpr       char& operator[](int i)       { return data[i]; }

  constexpr void push_back(char c) {
    data[n++] = c;
  }

  constexpr friend bool operator==(const Index& a, const Index& b) {
    if (a.n != b.n) {
      return false;
    }
    for (int i = 0; i < a.n; ++i) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  }

  constexpr friend bool operator<(const Index& a, const Index& b) {
    if (a.n < b.n) return true;
    if (b.n < a.n) return false;
    for (int i = 0; i < a.n; ++i) {
      if (a[i] < b[i]) return true;
      if (b[i] < a[i]) return false;
    }
    return false;
  }

  // Count the number of `c` in the index.
  constexpr int count(char c) const {
    return std::count(data, data + n, c);
  }

  // Return the index of the first instance of `c` in the index, or nullopt.
  constexpr std::optional<int> index_of(char c) const {
    if (auto i = std::find(data, data + n, c); i != data + n) {
      return i - data;
    }
    return std::nullopt;
  }

  // Hopefully obviously, search for chars in `search` and replace with the
  // corresponding char in `replace`.
  constexpr Index& search_and_replace(const Index& search, const Index& replace) {
    assert(search.n == replace.n);
    for (char& c : std::span(data)) {
      if (auto&& i = search.index_of(c)) {
        c = replace[*i];
      }
    }
    return *this;
  }
};

constexpr Index reverse(const Index& a) {
  Index out;
  for (int i = a.n - 1; i >= 0; --i) {
    out.push_back(a[i]);
  }
  return out;
}

constexpr Index unique(const Index& a) {
  Index out;
  for (char c : a) {
    if (out.count(c) == 0) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index repeated(const Index& a) {
  Index out;
  for (char c : a) {
    if (a.count(c) > 1 && !out.index_of(c)) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index exclusive(const Index& a) {
  Index out;
  for (char c : a) {
    if (a.count(c) == 1) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index& operator+=(Index& a, const Index& b) {
  for (char c : b) a.push_back(c);
  return a;
}

constexpr Index operator+(const Index& a, const Index& b) {
  return { a, b };
}

constexpr Index operator&(const Index& a, const Index& b) {
  Index out;
  for (char c : a) {
    if (b.index_of(c)) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index operator-(const Index& a, const Index& b) {
  Index out;
  for (char c : a) {
    if (!b.index_of(c)) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index operator^(const Index& a, const Index& b) {
  return (a - b) + (b - a);
}

constexpr bool permutation(const Index& a, const Index& b) {
  return (a - b).n == 0 && (b - a).n == 0;
}

constexpr std::string_view to_string(const Index& index) {
  return { index.begin(), index.end() };
}
}

template <>
struct fmt::formatter<ttl::Index> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::Index& index, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", to_string(index));
  }
};
