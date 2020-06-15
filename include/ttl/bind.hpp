#pragma once

#include "index.hpp"
#include "tensor.hpp"
#include <tuple>

namespace ttl {
template <Index Outer = decltype(index())>
class bind final {
  const tensor& a_;
  Outer outer_ = {};

 public:
  constexpr bind(const tensor& a) : a_(a) {
    assert(order(a) == 0);
  }

  constexpr bind(const tensor& a, Outer outer)
      : a_(a)
      , outer_(outer)
  {
    assert(order(a) == size(outer_));
  }

  friend constexpr std::string_view name(const bind&) {
    return "bind";
  }

  friend constexpr int order(const bind& b) {
    return size(unique(b.outer_));
  }

  friend constexpr decltype(auto) outer(const bind& b) {
    return unique(b.outer_);
  }

  friend constexpr decltype(auto) children(const bind& b) {
    return std::forward_as_tuple(b.a_);
  }

  template <Index Is>
  friend constexpr decltype(auto) rewrite(const bind& b, Is is) {
    assert(size(b.outer_) == size(is));
    return bind(b.a_, is);
  }

  template <Index... Is>
  constexpr decltype(auto) operator()(Is... is) const {
    return rewrite(*this, (is + ...));
  }
};

template <typename>    inline constexpr bool is_bind_v = false;
template <Index Outer> inline constexpr bool is_bind_v<bind<Outer>> = true;

template <typename T>
concept Bind = is_bind_v<std::remove_cvref_t<T>>;

template <Index... Is>
constexpr decltype(auto) tensor::operator()(Is... is) const {
  return bind(*this, is...);
}
}
