#pragma once

#include "Hessian.hpp"
#include "Tensor.hpp"
#include "Index.hpp"
#include "utils.hpp"

namespace ttl {
template <int N>
struct Scalar {
  Tensor         tensor;
  int         component = 0;
  std::array<int, N> dx = {};

  constexpr Scalar(Hessian h, std::same_as<int> auto... is)
      : tensor(h.tensor())
  {
    std::array<int, sizeof...(is)> index = { is... };
    Index outer = h.outer();

    for (int i = 0; auto&& c : h.index()) {
      component += utils::pow(N, i++) * index[*utils::index_of(outer, c)];
    }

    for (auto&& c : h.partial()) {
      ++dx[index[*utils::index_of(outer, c)]];
    }
  }

  constexpr bool operator==(const Scalar& rhs) {
    if (tensor != rhs.tensor) return false;
    if (component != rhs.component) return false;
    for (int n = 0; n < N; ++n) {
      if (dx[n] != rhs.dx[n]) return false;
    }
    return true;
  }

  constexpr bool operator<(const Scalar& rhs) const {
    if (partial_mask() < rhs.partial_mask()) return true;
    if (rhs.partial_mask() < partial_mask()) return false;
    if (id() < rhs.id()) return true;
    if (rhs.id() < id()) return false;
    if (component < rhs.component) return true;
    if (rhs.component < component) return false;
    for (int n = 0; n < N; ++n) {
      if (dx[n] < rhs.dx[n]) return true;
      if (rhs.dx[n] < dx[n]) return false;
    }
    return false;
  }

  constexpr std::string_view id() const {
    return tensor.id();
  }

  constexpr int partial_mask() const {
    int out = 0;
    for (int n = 0; n < N; ++n) {
      if (dx[n]) {
        out += utils::pow(2, n);
      }
    }
    return out;
  }

  std::string partial_string() const {
    constexpr const char ids[] = "xyz";
    std::string str;
    for (int n = 0; n < N; ++n) {
      for (int i = 0; i < dx[n]; ++i) {
        str += ids[n];
      }
    }
    return str;
  }
};
}
