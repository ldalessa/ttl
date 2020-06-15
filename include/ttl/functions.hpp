#pragma once

#include "delta.hpp"
#include "expression.hpp"
#include "index.hpp"
#include "operators.hpp"
#include "rational.hpp"

namespace ttl
{
template <Index A, Index B>
constexpr decltype(auto) delta(A&& i, B&& j) {
  return delta_node(std::forward<A>(i) + std::forward<B>(j));
}

template <Node A>
constexpr auto symmetrize(A&& a) {
  if constexpr (Tensor<A>) {
    return symmetrize(bind(std::forward<A>(a)));
  }
  else {
    return rational<1,2>() * (a + rewrite(a, reverse(outer(a))));
  }
}
}
