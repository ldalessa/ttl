#pragma once

#include <cassert>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>

namespace ttl::mp {
template <typename T, int M>
class cvector
{
 protected:
  T data[M] = {};
  int  m = 0;

  constexpr int check(int i) const {
    assert(0 <= i && i < m);
    return i;
  }

  constexpr int check_capacity(int i) const {
    assert(0 <= i && i < M);
    return i;
  }

 public:
  constexpr cvector() = default;

  constexpr cvector(std::same_as<T> auto... ds)
      : data { ds... }
      , m(sizeof...(ds))
  {
  }

  template <int A, int B>
  constexpr cvector(const cvector<T, A>& a, const cvector<T, B>& b) {
    for (const T& c : a) push(c);
    for (const T& c : b) push(c);
  }

  constexpr const T*  begin() const { return &data[0]; }
  constexpr       T*  begin()       { return &data[0]; }
  constexpr const T*    end() const { return &data[m]; }
  constexpr       T*    end()       { return &data[m]; }

  constexpr auto rbegin() const { return std::reverse_iterator(end()); }
  constexpr auto rbegin()       { return std::reverse_iterator(end()); }
  constexpr auto   rend() const { return std::reverse_iterator(begin()); }
  constexpr auto   rend()       { return std::reverse_iterator(begin()); }

  constexpr const T& operator[](int i) const { return data[check(i)]; }
  constexpr       T& operator[](int i)       { return data[check(i)]; }

  static constexpr int capacity()             { return M; }
  friend constexpr int size(const cvector& v) { return v.m; }

  constexpr void push(const T& t) { std::construct_at(&data[check_capacity(m++)], t); }
  constexpr void push(T&& t)      { std::construct_at(&data[check_capacity(m++)], std::move(t)); }

  template <typename... Ts>
  constexpr void emplace(Ts&&... ts) {
    std::construct_at(&data[check_capacity(m++)], std::forward<Ts>(ts)...);
  }

  constexpr T pop() {
    assert(m > 0);
    return std::move(data[--m]);
  }

  constexpr const T& back() const { return data[check(m - 1)]; }
  constexpr       T& back()       { return data[check(m - 1)]; }

  constexpr std::optional<int> find(const T& c) const {
    for (int i = 0; i < m; ++i) {
      if (data[i] == c) {
        return i;
      }
    }
    return std::nullopt;
  }

  constexpr int count(const T& c) const {
    int n = 0;
    for (int i = 0; i < m; ++i) {
      n += (data[i] == c);
    }
    return n;
  }

  constexpr friend std::true_type is_cvector_trait(cvector&) { return {}; }
};

cvector(auto... ds) -> cvector<std::common_type_t<decltype(ds)...>, sizeof...(ds)>;

template <typename T, int A, int B>
cvector(cvector<T, A>, cvector<T, B>) -> cvector<T, A + B>;

template <typename T>
concept is_cvector = requires (T t) {
  { is_cvector_trait(t)} -> std::same_as<std::true_type>;
};

/// Reverse the elements in a cvector.
template <is_cvector Vector>
constexpr Vector reverse(const Vector& a) {
  Vector out;
  for (auto i = std::rbegin(a), e = std::rend(a); i != e; ++i) {
    out.push(*i);
  }
  return out;
}

/// Return the unique elements in a cvector.
template <is_cvector Vector>
constexpr Vector unique(const Vector& a) {
  Vector out;
  for (const auto& c : a) {
    if (!out.find(c)) {
      out.push(c);
    }
  }
  return out;
}

/// Return the characters that only appear once in the cvector.
template <is_cvector Vector>
constexpr Vector exclusive(const Vector& a) {
  Vector out;
  for (const auto& c : a) {
    if (a.count(c) == 1) {
      out.push(c);
    }
  }
  return out;
}

/// Return the unique characters that appear more than once in the cvector.
template <is_cvector Vector>
constexpr Vector repeated(const Vector& a) {
  Vector out;
  for (const auto& c : a) {
    if (a.count(c) > 1) {
      if (!out.find(c)) {
        out.push(c);
      }
    }
  }
  return out;
}
}

