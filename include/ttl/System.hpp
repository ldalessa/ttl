#pragma once

#include "Equation.hpp"
#include <array>
#include <tuple>

namespace ttl
{
template <typename... Rhs> requires(is_tree<Rhs> && ...)
struct System
{
  static constexpr int M = sizeof...(Rhs);
  std::array<Tensor, M> lhs;
  std::tuple<Rhs...> rhs;

  constexpr System(is_equation auto&&... eqns) noexcept
      : lhs { eqns.lhs... }
      , rhs { eqns.rhs... }
  {
  }

  template <typename... Tuples>
  constexpr System(Tuples&&... tuples) noexcept
      : lhs {std::get<0>(tuples)...}
      , rhs {std::get<1>(tuples)...}
  {}

  constexpr std::array<std::string_view, M> tensors() const {
    std::array<std::string_view, M> out;
    std::transform(lhs.begin(), lhs.end(), out.begin(), [](auto t) {
      return t.id();
    });
    return out;
  }

  constexpr auto constants() const {
    constexpr int M = (Rhs::M + ... + 0);
    ce::cvector<std::string_view, M> out;

    auto search = [&](auto& tree) {
      for (int i = 0; i < tree.M; ++i) {
        if (auto* t = tree.at(i).tensor()) {
          if (std::find(lhs.begin(), lhs.end(), *t) == lhs.end()) {
            if (std::find(out.begin(), out.end(), t->id()) == out.end()) {
              out.push_back(t->id());
            }
          }
        }
      }
    };

    std::apply([&](auto&... rhs) {
      (search(rhs), ...);
    }, rhs);

    return out;
  }
};

template <typename... Tuples>
System(Tuples...) -> System<std::decay_t<std::tuple_element_t<1, Tuples>>...>;

template <typename... Rhs> requires(is_tree<Rhs> && ...)
System(Equation<Rhs>...) -> System<Rhs...>;

template <typename... Equations> requires(is_equation<Equations> && ...)
constexpr auto system(Equations&&... eqns) {
  return System(std::forward<Equations>(eqns)...);
}
}
