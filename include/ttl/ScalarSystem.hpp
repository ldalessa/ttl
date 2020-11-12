#pragma once

#include "Partial.hpp"
#include "System.hpp"
#include "ScalarTree.hpp"
#include "TensorTree.hpp"
#include <utility>

namespace ttl {
template <const auto& system, int N>
struct ScalarSystem
{
  constexpr static auto hessians = [] {
    constexpr int M = [] {
      auto hessians = system.hessians();
      return hessians.size();
    }();

    return []<std::size_t... m>(std::index_sequence<m...>) {
      auto h = system.hessians();
      return std::array{ h[m]... };
    }(std::make_index_sequence<M>());
  }();

  constexpr static auto constants = [] {
    constexpr int M = [] {
      auto constants = system.constants();
      return constants.size();
    }();

    std::array<const Tensor*, M> constants;
    for (int i = 0; auto c : system.constants()) {
      constants[i++] = c;
    }
    return constants;
  }();

  constexpr static auto simple = [] {
    // All of this is horrible but is in here to workaround
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=97790.
    constexpr auto simplify_all = [] {
      constexpr auto inner = []<std::size_t i>(std::integral_constant<std::size_t, i>) {
        return system.simplify(std::get<i>(system.trees()), constants);
      };

      return [&]<std::size_t... i>(std::index_sequence<i...>) {
        return std::tuple(inner(std::integral_constant<std::size_t, i>())...);
      }(std::make_index_sequence<system.size()>());
    };

    constexpr auto sizes = [&] {
      auto all = simplify_all();
      return std::apply([](auto const&... tree) {
        return std::array { tree.size()... };
      }, all);
    }();

    auto all = simplify_all();
    return [&]<std::size_t... i>(std::index_sequence<i...>) {
      return std::tuple(to_tree<TensorTree<sizes[i]>>(std::get<i>(all))...);
    }(std::make_index_sequence<system.size()>());
  }();

  constexpr static auto partials = [] {
    constexpr auto make_set = [] {
      utils::set<Partial<N>> out;
      out.reserve(64);
      for (auto&& h : hessians) {
        utils::expand(N, h.order(), [&](int index[]) {
          out.emplace(h, index);
        });
      }
      return out;
    };

    constexpr int M = [&] {
      auto partials = make_set();
      return partials.size();
    }();

    auto set = make_set();
    return PartialManifest<N, M>(std::move(set));
  }();

  constexpr static auto make_scalar_tree(is_tree auto const& tree) {
    return ScalarTree(tree, partials, constants);
  }

  constexpr static auto make_scalar_trees() {
    return std::apply([](is_tree auto const&... trees) {
      return std::tuple(make_scalar_tree(trees)...);
    }, simple);
  }
};

template <const auto& system, int N>
constexpr inline ScalarSystem<system, N> scalar_system = {};
}
