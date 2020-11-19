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
  constexpr static auto simple = [] {
    // All of this is horrible but is in here to workaround
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=97790.
    constexpr auto simplify_all = [] {
      constexpr auto inner = []<std::size_t i>(std::integral_constant<std::size_t, i>) {
        return system.simplify(std::get<i>(system.trees()), system.make_constants());
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

  constexpr static auto hessians = [] {
    constexpr auto make_set = [] {
      utils::set<Hessian> out;
      std::apply([&](auto const&... tree) {
        (system.hessians(out, tree), ...);
      }, simple);
      return out;
    };

    constexpr int M = [&] {
      auto set = make_set();
      return set.size();
    }();

    std::array<Hessian, M> hessians;
    for (int i = 0; auto&& h : make_set()) {
      hessians[i++] = h;
    }
    return hessians;
  }();

  constexpr static auto constants = [] {
    constexpr int M = [] {
      auto&& [constants, scalars] = system.make_partials(N);
      return constants.size();
    }();

    auto&& [constants, scalars] = system.make_partials(N);
    return PartialManifest<M>(N, std::move(constants));
  }();

  constexpr static auto scalars = [] {
    constexpr auto make_set = [] {
      utils::set<Partial> out;
      out.reserve(64);
      for (auto&& h : hessians) {
        if (not system.is_constant(h.tensor())) {
          utils::expand(N, h.order(), [&](int index[]) {
            out.emplace(N, h, index);
          });
        }
      }
      return out;
    };

    constexpr int M = [&] {
      auto scalars = make_set();
      return scalars.size();
    }();

    auto set = make_set();
    return PartialManifest<M>(N, std::move(set));
  }();

  constexpr static ScalarTree
  make_scalar_tree(is_tree auto const& tree)
  {
    return ScalarTree(N, tree, scalars, constants);
  }

  constexpr static auto make_scalar_trees() {
    ce::dvector<const ScalarTreeNode*> out;
    std::apply([&](is_tree auto const&... trees) {
      (ScalarTreeBuilder(N, trees, scalars, constants)(out), ...);
    }, simple);
    return out;
  }
};

template <const auto& system, int N>
constexpr inline ScalarSystem<system, N> scalar_system = {};
}
