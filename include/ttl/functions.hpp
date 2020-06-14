#pragma once

#include "delta.hpp"
#include "expression.hpp"
#include "index.hpp"
#include "operators.hpp"
#include "rational.hpp"

namespace ttl
{
constexpr delta_node delta(index<1> i, index<1> j) {
  return { i + j };
}

template <Expression A>
constexpr auto symmetrize(A a) {
  return rational<1,2>() * (a + rewrite(a, reverse(outer(a))));
}

constexpr auto symmetrize(const tensor& a) {
  return symmetrize(bind(a));
}
}
