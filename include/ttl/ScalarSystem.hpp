#pragma once

#include "System.hpp"

namespace ttl {
template <const auto& system, int N>
struct ScalarSystem
{
  constexpr static auto constants = [] {
    constexpr int M = [] { return system.constants().size(); }();
    std::array<const Tensor*, M> constants;
    for (int i = 0; auto c : system.constants()) {
      constants[i++] = c;
    }
    return constants;
  }();

  // constexpr static auto partials()
  // {
  //   // first pass computes the max number of scalars we might produce
  //   constexpr auto M = [] {
  //     int M = 0;
  //     for (auto&& h : hessians) {
  //       M += utils::pow(N, h.order());
  //     }
  //     return M;
  //   }();

  //   // second pass expands all of the hessians into their scalar partials
  //   constexpr auto partials = [&] {
  //     utils::set<Partial<N>, M> out;
  //     [&]<std::size_t... i>(std::index_sequence<i...>) {
  //       (utils::expand(N, order(hessians[i]), [&](int index[]) {
  //           out.emplace(hessians[i], index);
  //         }), ...);
  //     }(std::make_index_sequence<size(hessians)>());
  //     return out.sort();
  //   }();

  //   // finally truncate the result into a partial manifest
  //   return [&]<std::size_t... i>(std::index_sequence<i...>) {
  //     return PartialManifest(partials[i]...);
  //   }(std::make_index_sequence<size(partials)>());
  // }

  constexpr static auto simplify() {
    constexpr auto sizes = [] {
      return std::apply([](auto&&... tree) {
        return std::array<int, sizeof...(tree)>{size(tree)...};
      }, system.simplify());
    }();

    return [&]<std::size_t... i>(std::index_sequence<i...>) {
      return std::tuple([&]<typename T>(T&& tree) {
          return to_tree<Tree<sizes[i]>>(std::forward<T>(tree));
      }(system.simplify(rhs<i>(system)))...);
    }(std::make_index_sequence<sizes.size()>());
  }

  constexpr static auto simple = simplify();
};

template <const auto& system, int N>
constexpr inline ScalarSystem<system, N> scalar_system = {};
}
