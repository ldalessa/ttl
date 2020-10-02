#pragma once

#include "mp/cvector.hpp"
#include <ostream>
#include <string>
#include <string_view>

namespace ttl {
class Index : public mp::cvector<char, 8>
{
 public:
  using mp::cvector<char, 8>::cvector;

  constexpr Index& replace(const Index& is, const Index& with) {
    for (char& c : *this) {
      if (auto n = is.find(c)) {
        c = with[*n];
      }
    }
    return *this;
  }
};

constexpr std::string_view name(const Index& i) {
  return std::string_view(std::begin(i), std::end(i));
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

/// In-place set concatenation
constexpr Index& operator+=(Index& a, Index b) {
  for (char c : b) {
    a.push(c);
  }
  return a;
}

/// Set concatenation
constexpr Index operator+(Index a, Index b) {
  return a += b;
}

/// Set intersection
constexpr Index operator&(Index a, Index b) {
  Index out;
  for (char c : a) {
    if (b.find(c)) {
      out.push(c);
    }
  }
  return unique(out);
}

/// Set difference
constexpr Index operator-(Index a, Index b) {
  Index out;
  for (char c : a) {
    if (!b.find(c)) {
      out.push(c);
    }
  }
  return unique(out); // NB: is this really necessary?
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
