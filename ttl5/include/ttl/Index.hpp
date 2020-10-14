#pragma once

#include "ce/cvector.hpp"
#include <algorithm>
#include <ostream>
#include <string>
#include <string_view>

namespace ttl {
struct Index : public ce::cvector<char, 8>
{
  constexpr Index() = default;

  constexpr Index(std::same_as<char> auto... cs)
      : ce::cvector<char, 8>(std::in_place, cs...)
  {
  }

  constexpr Index& replace(const Index& is, const Index& with) {
    for (auto& c : *this) {
      if (auto i = std::find(is.begin(), is.end(), c); i != is.end()) {
        c = with[std::distance(is.begin(), i)];
      }
    }
    return *this;
  }

  constexpr std::string_view name() const {
    return { data(), data() + size() };
  }
};

constexpr std::string_view name(const Index& i) {
  return i.name();
}

std::ostream& operator<<(std::ostream& os, const Index& b) {
  return os << name(b);
}

constexpr bool operator==(const Index& a, const Index& b) {
  if (size(a) != size(b)) {
    return false;
  }
  for (int i = 0; i < size(a); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

/// Retain only unique indices in a.
constexpr Index unique(Index a) {
  Index out;
  std::copy_if(a.begin(), a.end(), std::back_inserter(out), [&](char c) {
    return std::find(out.begin(), out.end(), c) == out.end();
  });
  return out;
}

/// Retain the unique only indices that appear more than once in a.
constexpr Index repeated(Index a) {
  Index out;
  std::copy_if(a.begin(), a.end(), std::back_inserter(out), [&](char c) {
    return std::count(a.begin(), a.end(), c) > 1;
  });
  return unique(out);
}

/// Retain the indices that appear only once in a.
constexpr Index exclusive(Index a) {
  Index out;
  std::copy_if(a.begin(), a.end(), std::back_inserter(out), [&](char c) {
    return std::count(a.begin(), a.end(), c) == 1;
  });
  return out;
}

/// Set concatenation
constexpr Index operator+(Index a, Index b) {
  Index out;
  for (char c : a) out.push_back(c);
  for (char c : b) out.push_back(c);
  return out;
}

/// Set intersection
constexpr Index operator&(Index a, Index b) {
  Index out;
  std::copy_if(a.begin(), a.end(), std::back_inserter(out), [&](char c) {
    return std::find(b.begin(), b.end(), c) != b.end();
  });
  return unique(out);
}

/// Set difference
constexpr Index operator-(Index a, Index b) {
  Index out;
  std::copy_if(a.begin(), a.end(), std::back_inserter(out), [&](char c) {
    return std::find(b.begin(), b.end(), c) == b.end();
  });
  return out;
}

/// Set symmetric difference
constexpr Index operator^(Index a, Index b) {
  return (a - b) + (b - a);
}

constexpr int order(Index i) {
  return size(exclusive(i));
}

/// Check if two vectors are a permutation.
constexpr bool permutation(Index a, Index b) {
  return size(a - b) == 0 && size(b - a) == 0;
}

constexpr Index replace(const Index& is, const Index& with, Index in) {
  return in.replace(is, with);
}
}
