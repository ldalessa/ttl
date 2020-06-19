#pragma once

#include "expression.hpp"
#include "mp/ctad.hpp"

namespace ttl {
template <Expression A, Expression B>
class sum final {
  A a_;
  B b_;

 public:
  constexpr sum(A a, B b) : a_(a), b_(b) {
    auto ai = outer(a);
    auto bi = outer(b);
    assert(permutation(ai, bi));
  }

  friend constexpr std::string_view name(const sum&) {
    return "+";
  }

  friend constexpr int order(const sum& s) {
    return order(s.a_);
  }

  friend constexpr decltype(auto) outer(const sum& s) {
    return outer(s.a_);
  }

  friend constexpr decltype(auto) children(const sum& s) {
    return std::forward_as_tuple(s.a_, s.b_);
  }

  template <Index C>
  friend constexpr decltype(auto) rewrite(const sum& s, C index) {
    assert(order(s) == size(index));
    auto&& [l, r] = children(s);
    auto&&  o = outer(s);
    auto&& li = outer(l);
    auto&& ri = outer(r);
    return mp::ctad<sum>(rewrite(l, replace(o, index, li)), rewrite(r, replace(o, index, ri)));
  }

  template <Index... Is>
  constexpr decltype(auto) operator()(Is... is) const {
    return rewrite(*this, (is + ...));
  }
};

template <typename>                   inline constexpr bool is_sum_v = false;
template <Expression A, Expression B> inline constexpr bool is_sum_v<sum<A, B>> = true;
template <typename T> concept Sum = is_sum_v<std::remove_cvref_t<T>>;
}
