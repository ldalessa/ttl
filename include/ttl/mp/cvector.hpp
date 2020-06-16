#pragma once

#include <cassert>
#include <optional>
#include <ostream>
#include <type_traits>

namespace ttl::mp {
template <typename T, int N>
struct cvector {
  T data[N] = {};
  int n = 0;

  constexpr cvector() = default;

  template <typename... Is>
  constexpr cvector(Is... is) : data{is...} {}

  constexpr auto  begin() const { return data; }
  constexpr auto    end() const { return data + n; }
  constexpr auto rbegin() const { return std::reverse_iterator(end()); }
  constexpr auto   rend() const { return std::reverse_iterator(begin()); }

  constexpr decltype(auto) operator[](int i) const {
    assert(0 <= i && i < n);
    return data[i];
  }

  static constexpr int capacity() {
    return N;
  }

  constexpr int size() const {
    return n;
  }

  constexpr void push(int i) {
    assert(n < N);
    data[n++] = i;
  }

  template <typename U>
  constexpr int count(U&& u) const {
    int n = 0;
    for (auto&& t : *this) {
      n += (t == u);
    }
    return n;
  }

  template <typename U>
  constexpr std::optional<int> find(U&& u) const {
    for (int n = 0; auto&& t : *this) {
      if (t == u) {
        return n;
      }
      ++n;
    }
    return std::nullopt;
  }

  friend std::ostream& operator<<(std::ostream& os, const cvector& i) {
    for (auto&& t : i) {
      os << t;
    }
    return os;
  }

  // template <int M>
  // friend constexpr cvector<T, N+M> operator+(cvector a, cvector<T, M> b) {
  //   cvector<T, N+M> out;
  //   for (auto&& i : a) out.push(i);
  //   for (auto&& i : b) out.push(i);
  //   return out;
  // }
};

cvector() -> cvector<void, 0>;

template <typename T, typename... Ts>
cvector(T, Ts...) -> cvector<std::common_type_t<T, Ts...>, 1 + sizeof...(Ts)>;

template <typename>          inline constexpr bool is_cvector_v = false;
template <typename T, int N> inline constexpr bool is_cvector_v<cvector<T, N>> = true;
template <typename T>
concept CVector = is_cvector_v<std::remove_cvref_t<T>>;

template <typename Op, CVector Vector, typename... Is>
constexpr auto apply(Op&& op, Vector&& v, Is&&... is) {
  constexpr int N = Vector::capacity();
  constexpr int n = sizeof...(is);
  if constexpr (n < N) {
    if (n < v.size()) {
      return apply(std::forward<Op>(op), std::forward<Vector>(v), std::forward<Is>(is)..., v[n]);
    }
  }
  return op(std::forward<Is>(is)...);
}
}
