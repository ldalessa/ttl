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
};
}
