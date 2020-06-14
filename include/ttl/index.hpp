#pragma once

#include <cassert>
#include <ostream>
#include <optional>
#include <span>

namespace ttl {
template <int N>
struct index {
  char i[N] = {};
  int  n    = 0;

  constexpr index() = default;

  template <typename... Is>
  constexpr index(Is... is) : i{is...}, n(sizeof...(is)) {}

  static constexpr int capacity() {
    return N;
  }

  constexpr          auto  begin() const { return i; }
  constexpr          auto    end() const { return i + n; }
  constexpr decltype(auto) rbegin() const { return std::reverse_iterator(end()); }
  constexpr decltype(auto)   rend() const { return std::reverse_iterator(begin()); }

  constexpr decltype(auto) operator[](int j) const {
    assert(0 <= j && j < n);
    return i[j];
  }

  friend std::ostream& operator<<(std::ostream& os, const index& i) {
    for (auto&& c : i) {
      os << i;
    }
    return os;
  }

  friend constexpr int size(index i) {
    return i.n;
  }

  friend constexpr int capacity(index) {
    return N;
  }

  constexpr void push(char c) {
    assert(n != N);
    i[n++] = c;
  }

  constexpr int count(char c) const {
    int n = 0;
    for (auto&& a : *this) {
      n += (a == c);
    }
    return n;
  }

  constexpr std::optional<int> find(char c) const {
    for (int n = 0; auto&& a : *this) {
      if (c == a) {
        return n;
      }
      ++n;
    }
    return std::nullopt;
  }
};

template <typename... Is>
index(Is...) -> index<sizeof...(Is)>;

template <typename> inline constexpr bool is_index_v = false;
template <int N>    inline constexpr bool is_index_v<index<N>> = true;

template <typename T>
concept Index = is_index_v<T>;

template <int N>
constexpr index<N> reverse(index<N> a) {
  index<N> out;
  for (auto i = std::rbegin(a), e = std::rend(a); i != e; ++i) {
    out.push(*i);
  }
  return out;
}

template <int N>
constexpr index<N> unique(index<N> a) {
  index<N> out;
  for (auto&& c : a) {
    if (a.count(c) == 1) {
      out.push(c);
    }
  }
  return out;
}

template <int N>
constexpr index<N> repeated(index<N> a) {
  index<N> out;
  for (auto&& c : unique(a)) {
    if (a.count(c) > 1) {
      out.push(c);
    }
  }
  return out;
}

/// Set concatenation
template <int N, int M>
constexpr index<N+M> operator+(index<N> a, index<M> b) {
  index<N+M> out;
  for (auto&& c : a) out.push(c);
  for (auto&& c : b) out.push(c);
  return out;
}

/// Set intersection
template <int N, int M>
constexpr index<N+M> operator&(index<N> a, index<M> b) {
  index<N+M> out;
  for (auto&& c : a) {
    if (b.count(c)) {
      out.push(c);
    }
  }
  return out;
}

/// Set difference
template <int N, int M>
constexpr index<N> operator-(index<N> a, index<M> b) {
  index<N> out;
  for (auto&& c : a) {
    if (!b.count(c)) {
      out.push(c);
    }
  }
  return out;
}

/// Set symmetric difference
template <int N, int M>
constexpr index<N+M> operator^(index<N> a, index<M> b) {
  return (a - b) + (b - a);
}

template <int N, int M>
constexpr bool permutation(index<N> a, index<M> b) {
  return size(a - b) == 0 && size(b - a) == 0;
}

template <int N, int M, int O>
constexpr index<O> replace(index<N> is, index<M> with, index<O> in) {
  assert(size(is) == size(with));
  index<O> out;
  for (auto&& c : in) {
    if (auto n = is.find(c)) {
      out.push(with[*n]);
    }
    else {
      out.push(c);
    }
  }
  return out;
}
}
