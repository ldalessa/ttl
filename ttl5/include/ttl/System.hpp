#pragma once

#include "Tensor.hpp"
#include "ce/cvector.hpp"
#include <fmt/core.h>
#include <tuple>

namespace ttl {
template <typename... Trees>
struct System {
  static constexpr int M = sizeof...(Trees);
  std::reference_wrapper<const Tensor> lhs[M];
  std::tuple<const Trees&...>          rhs;

  template <typename... Tuples>
  requires((std::tuple_size_v<Tuples> == 2) && ...)
    constexpr System(Tuples&&... eqns)
    : lhs{std::ref(std::get<0>(eqns))...}
    , rhs{std::ref(std::get<1>(eqns))...}
  {
  }

  constexpr auto tensors() const {
  }

  constexpr auto constants() const {
    // Count the number of tensor nodes there are in the tree.
    constexpr int capacity = (Trees::count(TENSOR) + ...);

    ce::cvector<std::string_view, M> scalars;
    for (const auto& tensor : lhs) {
      scalars.push_back(name(tensor));
    }

    // Gather all of the tensor strings.
    ce::cvector<std::string_view, capacity> constants;

    auto gather = [&](const auto& tree) {
      tree.for_each([&](const Tensor& t) {
        if (std::find(scalars.begin(), scalars.end(), name(t)) == scalars.end())
        {
          constants.push_back(name(t));
        }
      });
    };

    std::apply([&](const auto&... trees) {
      (gather(trees), ...);
    }, rhs);

    std::sort(constants.begin(), constants.end());
    auto i = std::unique(constants.begin(), constants.end());
    constants.resize(std::distance(constants.begin(), i));
    return constants;
  }
};

template <typename... Tuples>
System(Tuples... eqns) -> System<std::remove_cvref_t<std::tuple_element_t<1, Tuples>>...>;

constexpr auto make_system_of_equations(auto&&... eqns) {
  return System(std::forward<decltype(eqns)>(eqns)...);
}
}
