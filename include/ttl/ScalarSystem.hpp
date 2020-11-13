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
        return system.simplify(std::get<i>(system.trees()), system.constants());
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
    constexpr auto make_set = [] {
      utils::set<Partial<N>> out;
      out.reserve(64);
      for (auto&& h : hessians) {
        if (system.is_constant(h.tensor())) {
          utils::expand(N, h.order(), [&](int index[]) {
            out.emplace(h, index);
          });
        }
      }
      return out;
    };

    constexpr int M = [&] {
      auto constants = make_set();
      return constants.size();
    }();

    auto set = make_set();
    return PartialManifest<N, M>(std::move(set));
  }();

  constexpr static auto scalars = [] {
    constexpr auto make_set = [] {
      utils::set<Partial<N>> out;
      out.reserve(64);
      for (auto&& h : hessians) {
        if (not system.is_constant(h.tensor())) {
          utils::expand(N, h.order(), [&](int index[]) {
            out.emplace(h, index);
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
    return PartialManifest<N, M>(std::move(set));
  }();

  constexpr static auto make_scalar_tree(is_tree auto const& tree) {
    return ScalarTree(tree, scalars, constants);
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
