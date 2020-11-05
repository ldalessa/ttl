#pragma once

#include "Tensor.hpp"
#include "Index.hpp"

namespace ttl {
struct Hessian {
  Tensor a_;
  Index dx_;
  Index  i_;

  constexpr Hessian(Tensor a, Index dx = {}, Index i = {})
      : a_(a)
      , dx_(dx)
      , i_(i)
  {
    // The hessians have anonymized indices, i.e., they only care about local
    // relative matching and not about external index identity. For example,
    // a(i,ji) and a(j,ij) are the same hessian.
    Index search = unique(i_ + dx_);
    Index replace;
    for (int i = 0, e = search.size(); i < e; ++i) {
      replace.push_back(char('0' + i));
    }
    i_.search_and_replace(search, replace);
    dx_.search_and_replace(search, replace);
  }

  constexpr bool operator==(const Hessian& rhs) {
    return a_ == rhs.a_ && i_ == rhs.i_ && dx_ == rhs.dx_;
  }

  constexpr bool operator<(const Hessian& rhs) {
    if (a_ < rhs.a_) return true;
    if (rhs.a_ < a_) return false;
    if (i_ < rhs.i_) return true;
    if (rhs.i_ < i_) return false;
    if (dx_ < rhs.dx_) return true;
    if (rhs.dx_ < dx_) return false;
    return false;
  }

  constexpr Tensor tensor() const {
    return a_;
  }

  constexpr Index index() const {
    return i_;
  }

  constexpr Index partial() const {
    return dx_;
  }

  constexpr Index inner() const {
    return index() + partial();
  }

  constexpr Index outer() const {
    return unique(inner());
  }

  constexpr int order() const {
    return outer().size();
  }

  constexpr friend int order(const Hessian& h) {
    return h.order();
  }
};
}
