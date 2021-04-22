#pragma once

#include "ttl/ExecutableTree.hpp"
#include "ttl/SerializedTree.hpp"
#include "ttl/Tag.hpp"
#include "ttl/TensorTree.hpp"
#include "ttl/TreeShape.hpp"
#include "kumi.hpp"
#include <array>

namespace ttl
{
  template <class T, int N, auto const& system>
  struct ExecutableSystem
  {
    constexpr static auto shapes = system.shapes(N);

    constexpr static auto serialize_trees()
    {
      return []<std::size_t... i>(std::index_sequence<i...>)
      {
        auto tensor_trees = system.simplify_trees();

        set<Scalar> all = tensor_trees([](is_tree auto const& ... tree) {
          set<Scalar> all;
          (tree.scalars(N, all), ...);
          return all;
        });

        all.sort();

        set<Scalar> constant_coefficients;
        set<Scalar> scalars;
        for (Scalar const& s : all) {
          if (s.constant) {
            constant_coefficients.emplace(s);
          }
          else {
            scalars.emplace(s);
          }
        }

        return kumi::make_tuple([&]
        {
          constexpr auto const& shape = kumi::get<i>(shapes);
          auto const& tree = kumi::get<i>(tensor_trees);
          return SerializedTree<T, shape>(tree, scalars, constant_coefficients);
        }()...);
      }(std::make_index_sequence<shapes.size()>());
    }

    constexpr static auto serialized_trees = serialize_trees();

    constexpr static auto make_executable_trees()
    {
      return []<std::size_t... i>(std::index_sequence<i...>)
      {
        return kumi::make_tuple([]
        {
          constexpr auto const& shape = kumi::get<i>(shapes);
          constexpr auto const&  tree = kumi::get<i>(serialized_trees);
          return ExecutableTree<T, shape, tree>();
        }()...);
      }(std::make_index_sequence<shapes.size()>());
    }

    constexpr static auto executable_trees = make_executable_trees();

    auto evaluate(auto const& scalars, auto const& constants) const {
      executable_trees([&](auto const&... tree) {
        (tree.evaluate(scalars, constants), ...);
      });
    }

    constexpr static set<Scalar> collect_scalars(bool constant)
    {
      set<Scalar> scalars;
      serialized_trees([&](auto const&... tree) {
        (tree.get_scalars(true, scalars), ...);
      });
      scalars.sort();
      return scalars;
    }

    constexpr static std::array constants = []
    {
      constexpr int M = collect_scalars(true).size();
      return to_array<M>(collect_scalars(true));
    }();

    constexpr static std::array scalars = []
    {
      constexpr int M = collect_scalars(false).size();
      return to_array<M>(collect_scalars(false));
    }();

  };
}
