#pragma once

#include "ScalarIndex.hpp"
#include "Tensor.hpp"
#include "pow.hpp"
#include <cassert>
#include <fmt/core.h>

namespace ttl
{
struct Scalar
{
  Tensor  tensor;
  int  component = 0;
  int       mask = 0;
  bool constant = false;
  ScalarIndex dx = {};

  constexpr Scalar() = default;

  constexpr Scalar(int N, const auto* node)
      : Scalar(N, node->tensor, node->index, node->constant)
  {
    assert(node->tag == TENSOR);
  }

  constexpr Scalar(int N, const Tensor& t, const ScalarIndex& index, bool constant)
      : tensor(t)
      , dx(N)
      , constant(constant)
  {
    assert(t.order() <= index.size());

    // first couple of indices select which component we're interacting with
    // base on the order of the tensor
    int i = 0;
    for (int e = t.order(); i < e; ++i) {
      component += pow(N, i) * index[i];
    }

    // the rest of the indices select which components of the higher order
    // hessian we're interacting with
    for (int e = index.size(); i < e; ++i) {
      ++dx[index[i]];
    }

    // build the partial mask
    for (int n = 0, e = dx.size(); n < e; ++n) {
      if (dx[n]) {
        mask += pow(2, n);
      }
    }

    assert(!constant || mask == 0);
  }

  constexpr friend bool operator==(const Scalar& a, const Scalar& b) {
    if (a.component != b.component) return false;
    if (a.tensor != b.tensor) return false;
    if (a.dx != b.dx) return false;
    return true;
  }

  // this ordering is important for the partial manifest
  constexpr friend bool operator<(const Scalar& a, const Scalar& b) {
    if (a.mask < b.mask) return true;
    if (b.mask < a.mask) return false;
    if (a.constant && !b.constant) return true;
    if (!b.constant && a.constant) return false;
    if (a.dx < b.dx) return true;
    if (b.dx < a.dx) return false;
    if (a.tensor.id() < b.tensor.id()) return true;
    if (b.tensor.id() < a.tensor.id()) return false;
    if (a.component < b.component) return true;
    if (b.component < a.component) return false;
    return false;
  }

  std::string to_string() const
  {
    constexpr static const char ids[] = "xyzw";

    std::string str;

    if (mask != 0) {
      str.append("d");
    }

    str.append(tensor.id());

    if (tensor.order()) {
      str.append(1, ids[component]);
    }

    if (mask == 0) {
      return str;
    }

    str.append("_d");

    for (int n = 0; n < dx.size(); ++n) {
      for (int i = 0; i < dx[n]; ++i) {
        str.append(1, ids[n]);
      }
    }
    return str;
  }
};
}

template <>
struct fmt::formatter<ttl::Scalar> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const ttl::Scalar& p, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", p.to_string());
  }
};
