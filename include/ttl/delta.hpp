#pragma once

#include "index.hpp"
#include "mp/ctad.hpp"
#include <string_view>
#include <type_traits>

namespace ttl {
/// @note For usability reasons this type is called a "delta_node" rather than a
///       "delta". This allows us to rely on ADL for the delta function in
///       functions.hpp.
template <Index Outer>
class delta_node final {
  Outer i_;
 public:
  constexpr delta_node(Outer i) : i_(i) {
    assert(size(i) == 2);
  }

  friend constexpr std::string_view name(const delta_node&) {
    return "delta";
  }

  friend constexpr int order(const delta_node&) {
    return 2;
  }

  friend constexpr decltype(auto) outer(const delta_node& d) {
    return d.i_;
  }

  template <Index A>
  friend constexpr decltype(auto) rewrite(const delta_node&, A index) {
    return mp::ctad<delta_node>(index);
  }

  template <Index A, Index B>
  constexpr decltype(auto) operator()(A&& a, B&& b) const {
    assert(size(a) == 1);
    assert(size(b) == 1);
    return rewrite(*this, std::forward<A>(a) + std::forward<B>(b));
  }
};

template <typename>    inline constexpr bool is_delta_v = false;
template <Index Outer> inline constexpr bool is_delta_v<delta_node<Outer>> = true;

template <typename A>
concept Delta = is_delta_v<std::remove_cvref_t<A>>;
}
