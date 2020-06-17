#pragma once

#include "expression.hpp"
#include "mp/cvector.hpp"
#include <tuple>

#include <iostream>

namespace ttl
{

template <typename Op, typename Reduce, LeafNode Leaf>
constexpr auto postfix(Op&& op, Reduce&& reduce, Leaf&& tree) {
  return op(std::forward<Leaf>(tree));
}

template <typename Op, typename Reduce, Internal Tree>
constexpr auto postfix(Op&& op, Reduce&& reduce, Tree&& tree) {
  return std::apply([&](auto&&... c) {
    return reduce(postfix(std::forward<Op>(op),
                          std::forward<Reduce>(reduce),
                          std::forward<decltype(c)>(c))...,
                  op(std::forward<Tree>(tree)));
  }, children(tree));
}

template <mp::CVector Keys, Node Tree>
constexpr mp::cvector<std::string_view, 0> constants(Keys&&, Tree&&) {
  return {};
}

template <mp::CVector Keys>
constexpr mp::cvector<std::string_view, 1> constants(Keys&& keys, const tensor& t) {
  if (keys.find(name(t))) {
    return {};
  }
  else {
    return { name(t) };
  }
}

template <mp::CVector Constants, Node Tree>
constexpr auto simplify(Constants&& constants, Tree&& tree) {

}

template <Cardinality<2>... Equations>
constexpr auto make_system_of_equations(Equations&&... eqns) {
  auto fields = mp::cvector(name(std::get<0>(std::forward<Equations>(eqns)))...);

  auto op = [&](auto&& tree) {
    return constants(fields, std::forward<decltype(tree)>(tree));
  };

  auto reduce = [](auto&&... tree) {
    return (std::forward<decltype(tree)>(tree) + ... + mp::cvector<std::string_view, 0>());
  };

  return reduce(postfix(op, reduce, std::get<1>(std::forward<Equations>(eqns)))...).unique();
}
}
