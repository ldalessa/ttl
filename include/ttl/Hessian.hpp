#pragma once

#include "Tensor.hpp"
#include "Index.hpp"

namespace ttl {
struct Hessian {
  Tensor a;
  Index  i;
  Index dx;

  constexpr Hessian(Tensor a, Index i, Index dx) : a(a), i(i), dx(dx) {}
  constexpr bool operator==(const Hessian& rhs) {
    return a == rhs.a && i == rhs.i && dx == rhs.dx;
  }

  constexpr bool operator<(const Hessian& rhs) {
    if (a < rhs.a) return true;
    if (rhs.a < a) return false;
    if (i < rhs.i) return true;
    if (rhs.i < i) return false;
    if (dx < rhs.dx) return true;
    if (rhs.dx < dx) return false;
    return false;
  }

  constexpr Tensor tensor() const {
    return a;
  }

  constexpr Index index() const {
    return i;
  }

  constexpr Index partial() const {
    return dx;
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
