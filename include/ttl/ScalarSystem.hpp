#pragma once

#include "Scalar.hpp"
#include "System.hpp"
#include "utils.hpp"

namespace ttl {
template <const auto& system, int N>
struct ScalarSystem {
  constexpr auto scalars() const {
    constexpr auto hessians = system.hessians();

    // first pass computes the max number of scalars we might produce.
    constexpr auto M = [&] {
      int M = 0;
      for (auto&& h : hessians) {
        M += utils::pow(N, h.order());
      }
      return M;
    }();

    // second pass generates the scalars.
    ce::cvector<Scalar<N>, M> out;
    [&]<auto... i>(utils::seq<i...>) {
      (expand<hessians[i].order()>([&](auto... is) {
        Scalar<N> s(hessians[i], is...);
        if (!utils::index_of(out, s)) {
          out.push_back(s);
        }
      }), ...);
    }(utils::make_seq_v<hessians.size()>);

    // sort the output so that partial masks are together
    std::sort(out.begin(), out.end());
    return out;
  }

  template <int Order, typename Op>
  constexpr static void expand(Op&& op, std::same_as<int> auto... is) {
    if constexpr (sizeof...(is) == Order) {
      op(is...);
    }
    else {
      for (int n = 0; n < N; ++n) {
        expand<Order>(std::forward<Op>(op), is..., n);
      }
    }
  }
};

template <const auto& system, int N>
constexpr inline ScalarSystem<system, N> scalar_system = {};
}
