/// Meta-programming utilities
#pragma once

#include <utility>

namespace ttl {
namespace utils {
template <typename Op, typename T, T... vs>
constexpr auto apply(Op&& op, std::integer_sequence<T, vs...>) {
  return std::forward<Op>(op)(std::integral_constant<T, vs>()...);
}

template <auto N, typename Op>
constexpr auto apply(Op&& op) {
  return apply(std::forward<Op>(op), std::make_integer_sequence<decltype(N), N>());
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
