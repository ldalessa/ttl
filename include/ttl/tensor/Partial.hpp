#pragma once

#include "ttl/Tensor.hpp"
#include "ttl/utils.hpp"
#include <cassert>

namespace ttl::tensor {
struct Partial
{
  Tensor tensor = {};
  int         N = 0;
  int max_alpha = 0;
  int component = 0;
  int        dx = 0;
  int      mask = 0;

  constexpr Partial() = default;

  constexpr Partial(int N, const Tensor& t, auto&& index)
      : tensor(t)
      , N(N)
  {
    assert(t.order() <= index.size());

    // first couple of indices select which component we're interacting with
    // base on the order of the tensor
    int i = 0;
    for (int e = t.order(); i < e; ++i) {
      component += utils::pow(N, i) * index[i];
    }

    // the rest of the indices select which components of the higher order
    // hessian we're interacting with
    ce::cvector<int> v(N);
    for (int e = index.size(); i < e; ++i) {
      ++v[index[i]];
    }

    // we now know what the maximum alpha of the partial derivative is, this is
    // used to encode dx in an integer
    for (int n = 0; n < N; ++n) {
      max_alpha = std::max(max_alpha, v[n]);
    }

    // encode the partial's mask and the actual partial dx as integers
    for (int n = 0; n < N; ++n) {
      if (v[n]) {
        mask += utils::pow(2, n);
        dx += utils::pow(max_alpha, n) * v[n];
      }
    }
  }

  constexpr friend bool operator==(const Partial& a, const Partial& b) {
    if (a.component != b.component) return false;
    if (a.max_alpha != b.max_alpha) return false;
    if (a.dx != b.dx) return false;
    if (a.tensor != b.tensor) return false;
    return true;
  }

  // this ordering is important for the partial manifest
  constexpr friend bool operator<(const Partial& a, const Partial& b) {
    if (a.mask < b.mask) return true;
    if (b.mask < a.mask) return false;
    if (a.max_alpha < b.max_alpha) return true;
    if (b.max_alpha < a.max_alpha) return false;
    if (a.dx < b.dx) return true;
    if (b.dx < a.dx) return false;
    if (a.tensor.id() < b.tensor.id()) return true;
    if (b.tensor.id() < a.tensor.id()) return false;
    if (a.component < b.component) return true;
    if (b.component < a.component) return false;
    return false;
  }
};
}
