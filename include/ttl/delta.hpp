#pragma once

#include "index.hpp"
#include <tuple>

namespace ttl {
/// @note For usability reasons this type is called a "delta_node" rather than a
///       "delta". This allows us to rely on ADL for the delta function in
///       functions.hpp.
class delta_node final {
  index<2> i_;
 public:
  constexpr delta_node(index<2> i) : i_(i) {
  }

  friend constexpr std::string_view name(const delta_node&) {
    return "delta";
  }

  friend constexpr int order(const delta_node&) {
    return 2;
  }

  friend constexpr index<2> outer(const delta_node& d) {
    return d.i_;
  }

  friend constexpr delta_node rewrite(const delta_node& d, index<2> is) {
    return delta_node(is);
  }

  constexpr decltype(auto) operator()(index<1> a, index<1> b) const {
    return rewrite(*this, a + b);
  }
};

template <typename A>
concept Delta = std::is_same_v<A, delta_node>;
}
