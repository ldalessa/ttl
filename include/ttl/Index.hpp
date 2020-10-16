#pragma once

#include <ce/cvector.hpp>
#include <fmt/format.h>
#include <algorithm>
#include <concepts>
#include <optional>

namespace ttl {
struct Index : ce::cvector<char, 8>
{
  constexpr Index() = default;
  constexpr Index(char c) : ce::cvector<char, 8>(std::in_place, c) {
  }

  // Count the number of `c` in the index.
  constexpr int count(char c) const {
    return std::count(begin(), end(), c);
  }

  // Return the index of the first instance of `c` in the index, or nullopt.
  constexpr std::optional<int> find(char c) const {
    if (auto i = std::find(begin(), end(), c); i != end()) {
      return std::distance(begin(), i);
    }
    return std::nullopt;
  }

  // Hopefully obviously, search for chars in `search` and replace with the
  // corresponding char in `replace`.
  constexpr Index& search_and_replace(Index search, Index replace) {
    assert(search.size() == replace.size());
    for (char& c : *this) {
      if (auto&& i = search.find(c)) {
        c = replace[*i];
      }
    }
    return *this;
  }

  constexpr std::string_view to_string() const {
    return { begin(), end() };
  }
};

constexpr Index reverse(Index a) {
  Index out;
  for (auto i = a.rbegin(), e = a.rend(); i != e; ++i) {
    out.push_back(*i);
  }
  return out;
}

constexpr Index unique(Index a) {
  Index out;
  for (char c : a) {
    if (out.count(c) == 0) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index repeated(Index a) {
  Index out;
  for (char c : a) {
    if (a.count(c) > 1 && !out.find(c)) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index exclusive(Index a) {
  Index out;
  for (char c : a) {
    if (a.count(c) == 1) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index operator+(Index a, Index b) {
  Index out;
  for (char c : a) out.push_back(c);
  for (char c : b) out.push_back(c);
  return out;
}


constexpr Index operator&(Index a, Index b) {
  Index out;
  for (char c : a) {
    if (b.find(c)) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index operator-(Index a, Index b) {
  Index out;
  for (char c : a) {
    if (!b.find(c)) {
      out.push_back(c);
    }
  }
  return out;
}

constexpr Index operator^(Index a, Index b) {
  return (a - b) + (b - a);
}

constexpr bool permutation(Index a, Index b) {
  return (a - b).size() == 0 && (b - a).size() == 0;
}
}

template <>
struct fmt::formatter<ttl::Index> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const ttl::Index& i, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", i.to_string());
  }
};
