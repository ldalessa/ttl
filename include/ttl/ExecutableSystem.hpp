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

    constexpr static auto make_executable_trees()
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
          constexpr int Scalars = shape.n_scalars;
          constexpr int Immediates = shape.n_immediates;
          using Tree = ExecutableTensorTree<T, N, Nodes, Indices, Scalars, Immediates>;
          return Tree(kumi::get<i>(tensor_trees), scalars, constant_coefficients);
        }()...);
      }(std::make_index_sequence<shapes.size()>());
    }

    constexpr static auto executable_trees = make_executable_trees();
  };
}
