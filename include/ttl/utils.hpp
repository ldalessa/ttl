/// Meta-programming utilities
#pragma once

#include <cassert>
#include <utility>

namespace ttl {
namespace utils {

template <int N = 0>
struct cvector {
  int data[N] = {};
  int n = 0;

  constexpr cvector() = default;

  template <typename... Is>
  constexpr cvector(Is... is) : data{is...} {}

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

  constexpr auto begin() const { return data; }
  constexpr auto   end() const { return data + n; }

  constexpr int operator[](int i) const {
    assert(i < n);
    return data[i];
  }

  template <int N, int M>
  friend constexpr cvector<N+M> operator+(cvector<N> a, cvector<M> b) {
    cvector<N+M> out;
    for (auto&& i : a) out.push(i);
    for (auto&& i : b) out.push(i);
    return out;
  }
};

template <typename... Is>
cvector(Is...) -> cvector<Is...>;

template <typename Op, typename T, T... vs>
constexpr auto apply(Op&& op, std::integer_sequence<T, vs...>) {
  return std::forward<Op>(op)(std::integral_constant<T, vs>()...);
}

template <auto N, typename Op>
constexpr auto apply(Op&& op) {
  return apply(std::forward<Op>(op), std::make_integer_sequence<decltype(N), N>());
}

template <typename Op, int N, typename... Is>
constexpr auto apply(Op&& op, cvector<N> v, Is&&... is) {
  constexpr int n = sizeof...(is);
  if constexpr (n < N) {
    if (n < v.size()) {
      return apply(std::forward<Op>(op), std::forward<Is>(is)..., v[n]);
    }
  }
  return op(std::forward<Is>(is)...);
}

/// This extends an index and forwards the index to the operator.
///
/// This is used during scalar traversals through the tensor tree. The parameter
/// N defines the dimensionality of the underlying space, while the parameter M
/// defines the upper bound on the order of the recursion.
///
/// When we're using the type index M and m are the same, but when we're using
/// the fixed and flexible index implementation they may differ. M prevents
/// infinite recursion.
template <int N, int M, typename Op, typename Ns, typename... Is>
constexpr auto extend(int m, Op&& op, Is... is) {
  if constexpr (M == 0) {
    assert(m == 0);
    return op(is...);
  }
  else {
    if (m == 0) {
      return op(is...);
    }
    else {
      return apply<N>([&](auto... n) {
        return (extend<N, M - 1>(op, m - 1, is..., n) + ...);
      });
    }
  }
}

/// A helper used inside a class template to trigger CTAD.
///
/// This function takes the name of a class template and constructs an instance
/// of that class template using the type discovered by class template argument
/// deduction.
///
/// This is used inside class templates in order to create instances of the
/// class template with different parameterized types.
///
/// ```
/// template <typename T>
/// struct Foo {
///   Foo(T) {}
///
///   template <typename U>
///   auto other(U u) const {
///     return Foo(u);        // <-- won't work because Foo means Foo<T>
///     return ctad<Foo>(u);  // <-- works
///   }
/// };
///
/// @tparam           T The class template to instantiate.
/// @tparam     Args... The constructor argument types.
///
/// @param      args... The constructor arguments.
///
/// @returns            A specialization of `T` as deduced by CTAD on `Args`.
template <template <typename...> typename T, typename... Args>
constexpr auto ctad(Args&&... args) {
  return T(std::forward<Args>(args)...);
}
}
}
