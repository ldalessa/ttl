#pragma once

#include "ExecutableTree.hpp"
#include "SIMDTree.hpp"
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

  static constexpr int n_scalars() {
    return M.scalars;
  }

  static constexpr int n_constants() {
    return M.constants;
  }

  static constexpr int n_trees() {
    return M.trees;
  }

  constexpr static auto constants = [] {
    constexpr int M = n_constants();
    return ScalarManifest<N, M>(system.scalars(N), true);
  }();

  constexpr static auto scalars = [] {
    constexpr int M = n_scalars();
    return ScalarManifest<N, M>(system.scalars(N), false);
  }();

  constexpr static auto executable = [] {
    constexpr std::array tree_sizes = [] {
      std::array<std::array<int, 2>, M.trees> out;
      auto trees = system.scalar_trees(N);
      for (int i = 0; i < M.trees; ++i) {
        out[i] = trees[i].size();
      }
      return out;
    }();

    auto&& trees = system.scalar_trees(N);
    return [&]<std::size_t... is>(std::index_sequence<is...>) {
      return std::tuple(ExecutableTree<tree_sizes[is][0], tree_sizes[is][1]>(trees[is], scalars, constants)...);
    }(std::make_index_sequence<n_trees()>());
  }();

  constexpr static auto simd = [] {
    constexpr int M = std::tuple_size_v<decltype(executable)>;

    // for each tree in the executable tuple
    //   generate a fully-typed eve tree by unpacking its tag geometry
    //
    // can't really do this cleanly because we need to maintain constexpr
    // context in the inner lambda

    return [&]<std::size_t... is>(std::index_sequence<is...>) {
      return std::tuple(
          [&]<std::size_t... js>(std::index_sequence<js...>) {
            constexpr auto&& tree = std::get<is>(executable);
            constexpr int Depth = tree.depth();
            return SIMDTree<Depth, tree.data[js].tag...>(tree);
          }(std::make_index_sequence<std::get<is>(executable).size()>())
          ...);
    }(std::make_index_sequence<M>());
  }();

  template <typename L, typename S, typename C>
  [[gnu::noinline]]
  constexpr static void evaluate(int n, L&& lhs, S&& scalars, C&& constants) {
    return std::apply([&](auto const&... e) {
      (e.evaluate(n, std::forward<L>(lhs), std::forward<S>(scalars), std::forward<C>(constants)), ...);
    }, executable);
  }

  template <typename L, typename S, typename C>
  [[gnu::noinline]]
  constexpr static void evaluate_simd(int n, L&& lhs, S&& scalars, C&& constants) {
    return std::apply([&](auto const&... e) {
      (e.evaluate(n, std::forward<L>(lhs), std::forward<S>(scalars), std::forward<C>(constants)), ...);
    }, simd);
  }
};
}
