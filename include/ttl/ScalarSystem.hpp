#pragma once

#include "Partial.hpp"
#include "System.hpp"
#include "TaggedTree.hpp"
#include "utils.hpp"

namespace ttl {
template <const auto& system, int N>
struct ScalarSystem
{
  constexpr static auto  hessians = system.hessians();
  constexpr static auto constants = system.constants();

  template <int M>
  constexpr friend auto rhs(ScalarSystem) {
    return std::get<M>(ScalarSystem::simple);
  }

  constexpr static auto partials()
  {
    // first pass computes the max number of scalars we might produce
    constexpr auto M = [] {
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

  constexpr static auto simplify_size(is_tree auto const& tree) {
    return size(system.simpify(tree, constants));
  }

  template <int M>
  constexpr static auto simplify_tags(is_tree auto const& tree) {
    std::array<Tag, M> tags;
    int i = 0;
    auto op = [&](const TreeNode* node, auto&& self) -> int {
      if (node->is_binary()) {
        self(node->a(), self);
        self(node->b(), self);
      }
      tags[i] = node->tag();
      return i++;
    };

    DynamicTree simple = system.simplify(tree);
    op(simple.root, op);
    return tags;
  }

  template <int M>
  constexpr static auto simplify_data(is_tree auto const& tree) {
    std::array<Node, M> data;
    int i = 0;
    auto op = [&](const TreeNode* node, auto&& self) -> int {
      if (node->is_binary()) {
        self(node->a(), self);
        self(node->b(), self);
      }
      data[i] = node->data();
      return i++;
    };

    DynamicTree simple = system.simplify(tree);
    op(simple.root, op);
    return data;
  }

  constexpr static auto simplify() {
    constexpr auto sizes = [] {
      return std::apply([](auto&&... tree) {
        return std::array<int, sizeof...(tree)>{size(tree)...};
      }, system.simplify());
    }();

    constexpr auto tags = [&] {
      return [&]<std::size_t... i>(std::index_sequence<i...>) {
        return std::tuple(simplify_tags<sizes[i]>(rhs<i>(system))...);
      }(std::make_index_sequence<size(system)>());
    }();

    auto data = [&]<std::size_t... i>(std::index_sequence<i...>) {
      return std::tuple(simplify_data<sizes[i]>(rhs<i>(system))...);
    }(std::make_index_sequence<size(system)>());

    return [&]<std::size_t... i>(std::index_sequence<i...>) {
      return std::tuple([&]<std::size_t... j>(auto n, std::index_sequence<j...>) {
          return RPNTree<std::get<n()>(tags)[sizes[n()] - j - 1]...>(std::get<n()>(data)[j]...);
        }(std::integral_constant<std::size_t, i>(), std::make_index_sequence<sizes[i]>())...);
    }(std::make_index_sequence<size(system)>());
  }

  constexpr static auto    simple = simplify();
};

template <const auto& system, int N>
constexpr inline ScalarSystem<system, N> scalar_system = {};
}
