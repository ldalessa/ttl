#pragma once

#include <cassert>
#include <utility>

namespace ttl {
namespace utils {
/// This extends an index and forwards the index to the operator.
///
/// This is used during scalar traversals through the tensor tree. The parameter
/// N defines the dimensionality of the underlying space, while the parameter M
/// defines the upper bound on the order of the recursion.
///
/// When we're using the type index M and m are the same, but when we're using
/// the fixed and flexible index implementation they may differ. M prevents
/// infinite recursion.
template <int N, int M, typename Op, typename Ns, typename... Is>
constexpr auto extend(int m, Op&& op, Is... is) {
  if constexpr (M == 0) {
    assert(m == 0);
    return op(is...);
  }
  else {
    if (m == 0) {
      return op(is...);
    }
    else {
      return apply<N>([&](auto... n) {
        return (extend<N, M - 1>(op, m - 1, is..., n) + ...);
      });
    }
  }
}
}
}
