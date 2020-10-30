#pragma once

#include "Tensor.hpp"
#include "Index.hpp"

namespace ttl {
struct Hessian {
  Tensor a;
  Index  i;
  Index  dx;

  constexpr Hessian(Tensor a, Index i, Index dx) : a(a), i(i), dx(dx) {}
  constexpr bool operator==(const Hessian& rhs) {
    return a == rhs.a && i == rhs.i && dx == rhs.dx;
  }

  // constexpr bool operator!=(const Hessian& rhs) {
  //   return a != rhs.a || i == rhs.i || dx == rhs.dx;
  // }
};
}
