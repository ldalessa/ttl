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
      std::array<std::array<int, 2>, M.trees> out;
      auto trees = system.scalar_trees(N);
      for (int i = 0; i < M.trees; ++i) {
        out[i] = trees[i]->size();
      }
      return out;
    }();

    auto&& trees = system.scalar_trees(N);
    return [&]<std::size_t... is>(std::index_sequence<is...>) {
      return std::tuple(ExecutableTree<tree_sizes[is][0], tree_sizes[is][1]>(trees[is], scalars, constants)...);
    }(std::make_index_sequence<M.trees>());
  }();

  constexpr static void eval(auto&& lhs, auto&& rhs, auto&& c, int i) {
    int j = 0;
    std::apply([&](const auto&... tree) {
      ([&] {
        utils::stack<double> stack;
        for (const auto& node : tree) {
          if (tag_is_binary(node.tag)) {
            double b = stack.pop();
            double a = stack.pop();
            double d = tag_apply(node.tag, a, b);
            stack.push(d);
          }
          else if (node.tag == TENSOR && node.constant) {
            stack.push(c(node.offset));
          }
          else if (node.tag == TENSOR) {
            stack.push(rhs(node.offest, i));
          }
          else if (node.tag == RATIONAL) {
            stack.push(to_double(node.q));
          }
          else {
            assert(node.tag == DOUBLE);
            stack.push(node.d);
          }
        }
        lhs(j++, i) = stack.pop();
        assert(stack.size() == 0);
      }(), ...);
    }, executable);
  }
};
}
