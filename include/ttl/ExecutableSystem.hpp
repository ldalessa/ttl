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
          // NB: would just use shape as a CNTTP once clang supports
          constexpr auto const& shape = kumi::get<i>(shapes);
          constexpr int Nodes = shape.n_nodes;
          constexpr int Indices = shape.n_indices;
          constexpr int InnerIndices = shape.n_inner_indices;
          constexpr int TensorIndices = shape.n_tensor_indices;
          constexpr int Scalars = shape.n_scalars;
          constexpr int Immediates = shape.n_immediates;
          constexpr int Stack = shape.stack_depth;
          using Tree = SerializedTensorTree<T, N, Nodes, Indices, InnerIndices, TensorIndices, Scalars, Immediates, Stack>;
          return Tree(shape, kumi::get<i>(tensor_trees), scalars, constant_coefficients);
        }()...);
      }(std::make_index_sequence<shapes.size()>());
    }

    constexpr static auto serialized_tensor_trees = serialize_tensor_trees();

    constexpr static auto make_executable_tensor_trees()
    {
      return []<std::size_t... i>(std::index_sequence<i...>)
      {
        return kumi::make_tuple([]
        {
          constexpr auto&&  tree = kumi::get<i>(serialized_tensor_trees);

          constexpr auto    tags = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<exec::Tag, tree.tags[j]...>();
          }(std::make_index_sequence<std::size(tree.tags)>());

          constexpr auto indices = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<char, tree.indices[j]...>();
          }(std::make_index_sequence<std::size(tree.indices)>());

          constexpr auto index_offsets = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<int, tree.index[j]...>();
          }(std::make_index_sequence<std::size(tree.index)>());

          constexpr auto inner_indices = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<char, tree.inner_indices[j]...>();
          }(std::make_index_sequence<std::size(tree.inner_indices)>());

          constexpr auto inner_index_offsets = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<int, tree.inner_index[j]...>();
          }(std::make_index_sequence<std::size(tree.inner_index)>());

          constexpr auto tensor_indices = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<char, tree.tensor_indices[j]...>();
          }(std::make_index_sequence<std::size(tree.tensor_indices)>());

          constexpr auto tensor_index_offsets = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<int, tree.tensor_index[j]...>();
          }(std::make_index_sequence<std::size(tree.tensor_index)>());

          constexpr auto scalar_ids = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<int, tree.scalar_ids[j]...>();
          }(std::make_index_sequence<std::size(tree.scalar_ids)>());

          constexpr auto scalar_ids_offsets = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<int, tree.scalar[j]...>();
          }(std::make_index_sequence<std::size(tree.scalar)>());

          constexpr auto left = [&]<std::size_t... j>(std::index_sequence<j...>) {
            return std::integer_sequence<int, tree.left[j]...>();
          }(std::make_index_sequence<std::size(tree.left)>());

          return ExecutableTensorTree(tree,
                                      tags,
                                      indices,
                                      index_offsets,
                                      inner_indices,
                                      inner_index_offsets,
                                      tensor_indices,
                                      tensor_index_offsets,
                                      scalar_ids,
                                      scalar_ids_offsets,
                                      left);
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
