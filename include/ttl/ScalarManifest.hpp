#pragma once

#include "Scalar.hpp"
#include "utils.hpp"
#include <span>

namespace ttl {
template <int N, int M>
struct ScalarManifest
{
  Scalar data[M];
  int bounds[utils::pow(2, N) + 1];

  constexpr ScalarManifest(utils::set<Scalar>&& scalars, bool constant)
  {
    for (int i = 0; const Scalar& s : scalars) {
      if (s.constant == constant) {
        data[i++] = s;
      }
    }

    std::sort(data, data + M);

    int i = 0;
    bounds[i] = 0;
    for (int m = 0; m < M; ++m) {
      if (data[m].mask != i) {
        bounds[++i] = m;
      }
    }

    for (; i < std::size(bounds); ++i) {
      bounds[i] = M;
    }
  }

  constexpr auto begin() const { return data; }
  constexpr auto   end() const { return data + M; }

  constexpr auto dx(int mask) const {
    return std::span(data + bounds[mask], data + bounds[mask + 1]);
  }

  constexpr int find(const Scalar& scalar) const {
    for (int i = 0; i < M; ++i) {
      if (scalar == data[i]) {
        return i;
      }
    }
    return M;
  }

  constexpr int find(const ScalarTree::Node* tree) const {
    return find({N, tree});
  }

  constexpr int find(const Tensor& t, std::integral auto... is) const {
    assert(t.order() == sizeof...(is));
    assert(((is < N) && ...));
    return find({N, t, {std::in_place, is...}, true});
  }

  constexpr const Scalar& operator[](int i) const { return data[i]; }

  // Makes client code prettier.
  constexpr int operator()(const Tensor& t, std::integral auto... is) const
  {
    return find(t, is...);
  }
};
}
