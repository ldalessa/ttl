#pragma once

#include "expression.hpp"
#include "mp/ctad.hpp"

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

  template <Index C>
  friend constexpr decltype(auto) rewrite(const product& p, C index) {
    assert(order(p) == size(index));
    auto&& [l, r] = children(p);
    auto&&  o = outer(p);
    auto&& li = outer(l);
    auto&& ri = outer(r);
    return mp::ctad<product>(rewrite(l, replace(o, index, li)), rewrite(r, replace(o, index, ri)));
  }

  template <Index... Is>
  constexpr decltype(auto) operator()(Is... is) const {
    return rewrite(*this, (is + ...));
  }

  // template <int N, typename Op, typename Is>
  // friend constexpr auto eval(const product& p, Op&& op, Is is) {
  //   assert(order(p) == size((is + ...)));
  //   auto&& oi = outer(p) + inner(p);
  //   constexpr int M = capacity_v<decltype(oi)>;
  //   return utils::apply([&](auto... i) {
  //     return utils::extend<N, M>(size(oi), [&](auto... is) {
  //       auto&& [l, r] = children(p);
  //       auto&& li = select(outer(l), oi, is...);
  //       auto&& ri = select(outer(r), oi, is...);
  //       return eval<N>(l, op, li) * eval<N>(r, op, ri);
  //     }, is...);
  //   }, is);
  // }
};

template <typename>                   inline constexpr bool is_product_v = false;
template <Expression A, Expression B> inline constexpr bool is_product_v<product<A, B>> = true;
template <typename T> concept Product = is_product_v<std::remove_cvref_t<T>>;
}
