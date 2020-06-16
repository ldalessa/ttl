#pragma once

#include "utils.hpp"
#include <cassert>
#include <ostream>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ttl
{
template <char i>
constexpr std::integral_constant<char, i> idx = {};

template <typename> inline constexpr bool is_index_v = false;
template <typename T> concept Index = is_index_v<std::remove_cvref_t<T>>;

namespace flexible {
template <int N>
struct index {
  char i[N] = {};
  int  n    = 0;

  constexpr index() = default;

  template <char i>
  constexpr index(std::integral_constant<char, i>) : i{i}, n(1) {}

  template <typename... Is>
  constexpr index(Is... is) : i{is...}, n(sizeof...(is)) {}

  constexpr auto  begin() const { return i; }
  constexpr auto    end() const { return i + n; }
  constexpr auto rbegin() const { return std::reverse_iterator(end()); }
  constexpr auto   rend() const { return std::reverse_iterator(begin()); }

  constexpr char operator[](int j) const {
    assert(0 <= j && j < n);
    return i[j];
  }

  friend std::ostream& operator<<(std::ostream& os, const index& i) {
    for (auto&& c : i) {
      os << i;
    }
    return os;
  }

  static constexpr int capacity() {
    return N;
  }

  friend constexpr int size(index i) {
    return i.n;
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

template <char i>
index(std::integral_constant<char, i>) -> index<1>;

template <typename... Is>
index(Is...) -> index<sizeof...(Is)>;

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
} // namespace flexible

template <int N>
inline constexpr bool is_index_v<flexible::index<N>> = true;

inline
namespace fixed {
struct index {
  static constexpr int N = 8;

  char i[N] = {};
  int  n    = 0;

  constexpr index() = default;

  template <char idx>
  constexpr index(std::integral_constant<char, idx>) : n(1) {
    i[0] = idx;
  }

  template <typename... Is>
  constexpr index(Is... is) : i{is...}, n(sizeof...(is)) {}

  constexpr auto  begin() const { return i; }
  constexpr auto    end() const { return i + n; }
  constexpr auto rbegin() const { return std::reverse_iterator(end()); }
  constexpr auto   rend() const { return std::reverse_iterator(begin()); }

  constexpr char operator[](int j) const {
    assert(0 <= j && j < n);
    return i[j];
  }

  friend std::ostream& operator<<(std::ostream& os, const index& i) {
    for (auto&& c : i) {
      os << i;
    }
    return os;
  }

  static constexpr int capacity() {
    return N;
  }

  friend constexpr int size(index i) {
    return i.n;
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

constexpr index reverse(index a) {
  index out;
  for (auto i = std::rbegin(a), e = std::rend(a); i != e; ++i) {
    out.push(*i);
  }
  return out;
}

constexpr index unique(index a) {
  index out;
  for (auto&& c : a) {
    if (a.count(c) == 1) {
      out.push(c);
    }
  }
  return out;
}

constexpr index repeated(index a) {
  index out;
  for (auto&& c : unique(a)) {
    if (a.count(c) > 1) {
      out.push(c);
    }
  }
  return out;
}

/// Set concatenation
constexpr index operator+(index a, index b) {
  index out;
  for (auto&& c : a) out.push(c);
  for (auto&& c : b) out.push(c);
  return out;
}

/// Set intersection
constexpr index operator&(index a, index b) {
  index out;
  for (auto&& c : a) {
    if (b.count(c)) {
      out.push(c);
    }
  }
  return out;
}

/// Set difference
constexpr index operator-(index a, index b) {
  index out;
  for (auto&& c : a) {
    if (!b.count(c)) {
      out.push(c);
    }
  }
  return out;
}

/// Set symmetric difference
constexpr index operator^(index a, index b) {
  return (a - b) + (b - a);
}

constexpr bool permutation(index a, index b) {
  return size(a - b) == 0 && size(b - a) == 0;
}

constexpr index replace(index is, index with, index in) {
  assert(size(is) == size(with));
  index out;
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
} // namespace fixed

template <>
inline constexpr bool is_index_v<fixed::index> = true;

//inline
namespace typed {
template <char... Is>
struct index {
  static constexpr flexible::index<sizeof...(Is)> impl = { Is... };

  constexpr index() = default;

  template <char i>
  constexpr index(std::integral_constant<char, i>) {}

  constexpr auto begin() const { return impl.begin(); }
  constexpr auto   end() const { return impl.end(); }

  static constexpr int capacity() {
    return sizeof...(Is);
  }
};

template <char i>
index(std::integral_constant<char, i>) -> index<i>;

template <char... A>
constexpr int size(index<A...>) {
  return sizeof...(A);
}

template <char... A>
constexpr auto reverse(index<A...> a) {
  constexpr auto b = reverse(a.impl);
  return utils::apply<size(b)>([=](auto... i) {
    return index<b[i]...>();
  });
}

template <char... A>
constexpr auto unique(index<A...> a) {
  constexpr auto b = unique(a.impl);
  return utils::apply<size(b)>([=](auto... i) {
    return index<b[i]...>();
  });
}

template <char... A>
constexpr auto repeated(index<A...> a) {
  constexpr auto b = repeated(a.impl);
  return utils::apply<size(b)>([=](auto... i) {
    return index<b[i]...>();
  });
}

/// Set concatenation
template <char... A, char... B>
constexpr index<A..., B...> operator+(index<A...>, index<B...>) {
  return {};
}

/// Set intersection
template <char... A, char... B>
constexpr auto operator&(index<A...> a, index<B...> b) {
  constexpr auto c = a.impl & b.impl;
  return utils::apply<size(c)>([=](auto... i) {
    return index<c[i]...>();
  });
}

/// Set difference
template <char... A, char... B>
constexpr auto operator-(index<A...> a, index<B...> b) {
  constexpr auto c = a.impl - b.impl;
  return utils::apply<size(c)>([=](auto... i) {
    return index<c[i]...>();
  });
}

/// Set symmetric difference
template <char... A, char... B>
constexpr auto operator^(index<A...> a, index<B...> b) {
  return (a - b) + (b - a);
}

template <char... A, char... B>
constexpr bool permutation(index<A...> a, index<B...> b) {
  return size(a - b) == 0 && size(b - a) == 0;
}

template <char... A, char... B, char... C>
constexpr auto replace(index<A...> is, index<B...> with, index<C...> in) {
  constexpr auto d = replace(is.impl, with.impl, in.impl);
  return utils::apply<size(d)>([=](auto... i) {
    return index<d[i]...>();
  });
}
} // namespace fixed

template <char... A>
inline constexpr bool is_index_v<typed::index<A...>> = true;

template <Index A>
inline constexpr int capacity_v = A::capacity();
}
