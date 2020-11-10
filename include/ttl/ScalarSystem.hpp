#pragma once

#include "Partial.hpp"
#include "System.hpp"
#include "TensorTree.hpp"
#include <utility>

namespace ttl {
template <const auto& system, int N>
struct ScalarSystem
{
  constexpr static auto hessians = [] {
    constexpr int M = [] { return system.hessians().size(); }();
    return []<std::size_t... m>(std::index_sequence<m...>) {
      auto h = system.hessians();
      return std::array{ h[m]... };
    }(std::make_index_sequence<M>());
  }();

  constexpr static auto constants = [] {
    constexpr int M = [] { return system.constants().size(); }();
    std::array<const Tensor*, M> constants;
    for (int i = 0; auto c : system.constants()) {
      constants[i++] = c;
    }
    return constants;
  }();

  constexpr static auto simple = [] {
    constexpr auto sizes = [] {
      return std::apply([](auto&&... tree) {
        return std::array<int, sizeof...(tree)>{size(tree)...};
      }, system.simplify());
    }();

    auto trees = system.simplify();

    return [&]<std::size_t... i>(std::index_sequence<i...>) {
      return std::tuple([&]<typename T>(T&& tree) {
        return to_tree<TensorTree<sizes[i]>>(std::forward<T>(tree));
      }(system.simplify(rhs<i>(system)))...);
    }(std::make_index_sequence<sizes.size()>());
  }();

  constexpr static auto partials = [] {
    auto build = [] {
      utils::set<Partial<N>> out;
      for (auto&& h : hessians) {
        utils::expand(N, h.order(), [&](int index[]) {
          out.emplace(h, index);
        });
      }
      return out;
    };

    constexpr int M = [&] { return build().size(); }();
    return PartialManifest<N, M>(build());
  }();
};

template <const auto& system, int N>
constexpr inline ScalarSystem<system, N> scalar_system = {};
}
