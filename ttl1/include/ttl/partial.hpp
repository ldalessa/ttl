#pragma once

#include "expression.hpp"
#include "index.hpp"
#include "mp/ctad.hpp"

namespace ttl {
template <Expression A, Index Direction>
class partial final {
  A a_;
  Direction dx_;

 public:
  constexpr partial(A a, Direction dx) : a_(a), dx_(dx) {}

  friend constexpr std::string_view name(const partial&) {
    return "dx";
  }

  friend constexpr int order(const partial& p) {
    return size(outer(p));
  }

  friend constexpr decltype(auto) outer(const partial& p) {
    return unique(outer(p.a_) + p.dx_);
  }

  friend constexpr decltype(auto) children(const partial& p) {
    return std::forward_as_tuple(p.a_);
  }

  template <Index B>
  friend constexpr decltype(auto) rewrite(const partial& p, B index) {
    assert(order(p) == size(index));
    auto&&  o = outer(p);
    auto&& ai = outer(p.a_);
    auto&& pi = p.dx_;
    return mp::ctad<partial>(rewrite(p.a_, replace(o, index, ai)), replace(o, index, pi));
  }

  template <Index... Is>
  constexpr decltype(auto) operator()(Is... is) const {
    return rewrite(*this, (is + ...));
  }

  template <Index... Is>
  constexpr decltype(auto) append(Is... is) const {
    return mp::ctad<partial>(a_, (dx_ + ... + is));
  }
};

template <typename>
inline constexpr bool is_partial_v = false;

template <Expression A, Index Direction>
inline constexpr bool is_partial_v<partial<A, Direction>> = true;

template <typename T>
concept Partial = is_partial_v<std::remove_cvref_t<T>>;
}
