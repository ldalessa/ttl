#pragma once

#include "expression.hpp"

namespace ttl {
template <Expression A, Expression B>
class product final {
  A a_;
  B b_;

 public:
  constexpr product(A a, B b) : a_(a), b_(b) {
  }

  friend constexpr std::string_view name(const product&) {
    return "*";
  }

  friend constexpr int order(const product& p) {
    return size(outer(p));
  }

  friend constexpr decltype(auto) outer(const product& p) {
    return outer(p.a_) ^ outer(p.b_);
  }

  friend constexpr decltype(auto) children(const product& p) {
    return std::forward_as_tuple(p.a_, p.b_);
  }

  template <Index Is>
  friend constexpr decltype(auto) rewrite(const product& p, Is is) {
    assert(order(p) == size(is));
    auto&& [l, r] = children(p);
    auto&&  o = outer(p);
    auto&& li = outer(l);
    auto&& ri = outer(r);
    return product(rewrite(l, replace(o, is, li)), rewrite(r, replace(o, is, ri)));
  }

  template <Index... Is>
  constexpr decltype(auto) operator()(Is... is) const {
    return rewrite(*this, (is + ...));
  }
};

template <typename>                   inline constexpr bool is_product_v = false;
template <Expression A, Expression B> inline constexpr bool is_product_v<product<A, B>> = true;
template <typename T> concept Product = is_product_v<std::remove_cvref_t<T>>;
}
