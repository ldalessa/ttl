#pragma once

#include "ttl/ExecutableTensorTree.hpp"
#include "ttl/Tag.hpp"
#include "ttl/TensorTree.hpp"
#include "ttl/TreeShape.hpp"
#include "kumi.hpp"
#include <array>

namespace ttl
{
  template <auto const& system, class T, int N>
  struct ExecutableSystem
  {
    constexpr static auto shapes = system.shapes(N);

    constexpr static auto serialize_tensor_trees()
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
          return SerializedTensorTree<shape>(tree, scalars, constant_coefficients);
        }()...);
      }(std::make_index_sequence<shapes.size()>());
    }

    constexpr static auto serialized_tensor_trees = serialize_tensor_trees();

    constexpr static auto make_executable_tensor_trees()
    {
      return []<std::size_t... i>(std::index_sequence<i...>)
      {
        auto tensor_trees = system.simplify_trees();
        return kumi::make_tuple([&]
        {
          constexpr auto const& shape = kumi::get<i>(shapes);
          constexpr auto const&  tree = kumi::get<i>(serialized_tensor_trees);
          return ExecutableTensorTree<T, shape, tree>(kumi::get<i>(tensor_trees));
        }()...);
      }(std::make_index_sequence<shapes.size()>());
    }

    constexpr static auto executable_trees = make_executable_tensor_trees();

    auto evaluate(auto const& scalars, auto const& constants) const {
      executable_trees([&](auto const&... tree) {
        (tree.evaluate(scalars, constants), ...);
      });
    }
  };
}
