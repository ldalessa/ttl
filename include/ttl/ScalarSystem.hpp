#pragma once

#include "ScalarManifest.hpp"

namespace ttl {
template <auto const& system, int N>
struct ScalarSystem
{
  constexpr static int dim() { return N; }

  constexpr static auto M = [] {
    struct {
      int constants = 0;
      int scalars = 0;
      constexpr int operator()() const { return constants + scalars; }
    } out;

    for (auto&& s: system.scalars(N)) {
      out.constants += s.constant;
      out.scalars += !s.constant;
    }

    return out;
  }();

  constexpr static auto constants = [] {
    return ScalarManifest<N, M.constants>(system.scalars(N), true);
  }();

  constexpr static auto scalars = [] {
    return ScalarManifest<N, M.scalars>(system.scalars(N), false);
  }();
};
}
