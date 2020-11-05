#pragma once

#include "Partial.hpp"
#include "System.hpp"
#include "utils.hpp"

namespace ttl {
template <const auto& system, int N>
struct ScalarSystem
{
  constexpr static auto  hessians = system.hessians();
  constexpr static auto constants = system.constants();

  constexpr static auto partials()
  {
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
        (utils::expand(N, order(hessians[i]), [&](int index[]) {
            out.emplace(hessians[i], index);
          }), ...);
      }(std::make_index_sequence<size(hessians)>());
      return out.sort();
    }();

    // finally truncate the result into a partial manifest
    return [&]<std::size_t... i>(std::index_sequence<i...>) {
      return PartialManifest(partials[i]...);
    }(std::make_index_sequence<size(partials)>());
  }

  template <typename Tree>
  constexpr static Tree simplify(const Tree& tree) {
    utils::stack<int> constant;
    for (int i = 0; i < Tree::size(); ++i) {
      if (tree.at(i).is_binary()) {
        constant.push(constant.pop() & constant.pop());
      }
      else if (const Tensor* t = tree.at(i).tensor()) {
        constant.push(constants.contains(*t));
      }
      else {
        constant.push(true);
      }
    }
    return tree;
  }

  constexpr static auto simplify() {
    []<std::size_t... is>(std::index_sequence<is...>) {
      ([] {
        constexpr auto& tree = rhs<is>(system);
        constexpr int M = size(tree);
        constexpr auto copy = simplify(tree);
      }, ...);
    }(std::make_index_sequence<size(system)>());
  }
};

template <const auto& system, int N>
constexpr inline ScalarSystem<system, N> scalar_system = {};
}
