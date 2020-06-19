#pragma once

#include "expression.hpp"

namespace ttl {
template <Expression A>
class inverse final {
  A a_;
 public:
  constexpr inverse(A a) : a_(a) {
  }

  friend constexpr std::string_view name(const inverse&) {
    return "inverse";
  }

  friend constexpr int order(const inverse& i) {
    return order(i.a_);
  }

  friend constexpr decltype(auto) outer(const inverse& i) {
    return outer(i.a_);
  }

  friend constexpr decltype(auto) children(const inverse& i) {
    return std::forward_as_tuple(i.a_);
  }

  template <Index Is>
  friend constexpr decltype(auto) rewrite(const inverse& i, Is is) {
    assert(order(i) == size(is));
    return inverse(rewrite(i.a_, is));
  }

  template <Index... Is>
  constexpr decltype(auto) operator()(Is... is) const {
    return rewrite(*this, (is + ...));
  }
};

template <typename>
inline constexpr bool is_inverse_v = false;

template <Expression A>
inline constexpr bool is_inverse_v<inverse<A>> = true;

template <typename T>
concept Inverse = is_inverse_v<std::remove_cvref_t<T>>;
}
