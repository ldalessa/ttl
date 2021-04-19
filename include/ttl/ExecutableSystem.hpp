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
          std::array<T, shape.n_immediates> immediates;
          auto st = SerializedTree<T, shape>(tree, immediates, scalars, constant_coefficients);
          return kumi::make_tuple(st, immediates);
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
          constexpr auto const&       shape = kumi::get<i>(shapes);
          constexpr auto const&        tree = kumi::get<0>(kumi::get<i>(serialized_trees));
          constexpr auto const&  immediates = kumi::get<1>(kumi::get<i>(serialized_trees));
          return ExecutableTree<T, shape, tree>(immediates);
        }()...);
      }(std::make_index_sequence<shapes.size()>());
    }

    constexpr static auto executable_trees = make_executable_trees();

    auto evaluate(auto const& scalars, auto const& constants) const {
      executable_trees([&](auto const&... tree) {
        (tree.evaluate(scalars, constants), ...);
      });
    }
  };
}
