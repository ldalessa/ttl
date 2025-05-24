#pragma once

#include "ttl/ExecutableTree.hpp"
#include "ttl/SerializedTree.hpp"
#include "ttl/Tag.hpp"
#include "ttl/TensorTree.hpp"
#include "ttl/TreeShape.hpp"
#include <kumi/tuple.hpp>
#include <array>
#include <cstdio>
#include <format>
#include <bitset>

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
        (tree.get_scalars(constant, scalars), ...);
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

    /// Take a set of user-bound scalar constants and turn them into an array
    /// suitable for evaluate().
    ///
    /// It is okay to provide a bound scalar that doesn't exist in the problem
    /// declaration, it will just be ignored. This can happen if the source code
    /// is written to support multiple dimensionalities.
    ///
    /// It is okay to provide multiple definitions of the same scalar, only the
    /// first one will be used.
    ///
    /// All of the scalars in the problem must be provided at the same time.
    constexpr static auto map_constants(kumi::product_type auto... tuples)
    {
      constexpr int M = constants.size();
      using Tuple = kumi::tuple<Scalar, double>;
      static_assert((std::same_as<Tuple, decltype(tuples)> && ...));
      std::array<Tuple, M> out;
      auto begin = constants.begin();
      auto end = constants.end();
      std::bitset<M> bits;
      ([&] {
        auto scalar = kumi::get<0>(tuples);
        scalar.constant = true;
        if (!scalar.validate(N)) {
          if (!std::is_constant_evaluated()) {
            std::printf("%s", std::format("ignoring impossible constant: {}\n", scalar).c_str());
          }
          return;
        }

        if (auto i = std::find(begin, end, scalar); i != end) {
          int n = i - begin;
          if (!bits.test(n)) {
            out[n] = tuples;
            bits.set(n);
          }
          else {
            std::printf("%s", std::format("ignoring multiply specified constant: {}\n", scalar).c_str());
          }
        }
        else {
          std::printf("%s", std::format("invalid constant: {}\n", scalar).c_str());
        }
      }(), ...);

      if (bits.count() != M) {
        for (int i = 0; i < M; ++i) {
          if (!bits[i]) {
            if (!std::is_constant_evaluated()) {
              std::printf("%s", std::format("missing constant: {}\n", constants[i]).c_str());
            }
          }
        }
        assert(false);
      }

      return out;
    }
  };
}
