#pragma once

#include "Partial.hpp"
#include "System.hpp"
#include "utils.hpp"

namespace ttl {
template <const auto& system, int N>
struct ScalarSystem {
  constexpr static auto partials() {
    constexpr auto hessians = system.hessians();

    // first pass computes the max number of scalars we might produce
    constexpr auto M = [&] {
      int M = 0;
      for (auto&& h : hessians) {
        M += utils::pow(N, h.order());
      }
      return M;
    }();

    // second pass expands all of the hessians into their scalar partials
    constexpr auto partials = [&] {
      utils::set<Partial<N>, M> out;
      [&]<std::size_t... i>(std::index_sequence<i...>) {
        (utils::expand(N, order(hessians[i]), [&]<typename Index>(Index&& index) {
            out.emplace(hessians[i], std::forward<Index>(index));
          }), ...);
      }(std::make_index_sequence<size(hessians)>());
      return out.sort();
    }();

    // finally truncate the result into an array
    return [&]<std::size_t... i>(std::index_sequence<i...>) {
      return utils::make_array<Partial<N>>(partials[i]...);
    }(std::make_index_sequence<size(partials)>());
  }
};

template <const auto& system, int N>
constexpr inline ScalarSystem<system, N> scalar_system = {};
}
