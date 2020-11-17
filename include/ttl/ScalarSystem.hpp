#pragma once

#include "ExecutableTree.hpp"
#include "ScalarManifest.hpp"
#include <array>

namespace ttl {
template <auto const& system, int N>
struct ScalarSystem
{
  constexpr static int dim() {
    return N;
  }

  constexpr static auto M = [] {
    struct {
      int constants = 0;
      int scalars = 0;
      int trees = system.n_scalar_trees(N);
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

  constexpr static auto executable = [] {
    constexpr std::array tree_sizes = [] {
      std::array<int, M.trees> out;
      auto trees = system.scalar_trees(N);
      for (int i = 0; i < M.trees; ++i) {
        out[i] = trees[i]->size();
      }
      return out;
    }();

    auto&& trees = system.scalar_trees(N);
    return [&]<std::size_t... is>(std::index_sequence<is...>) {
      return std::tuple(ExecutableTree<tree_sizes[is]>(trees[is], scalars, constants)...);
    }(std::make_index_sequence<M.trees>());
  }();
};
}
