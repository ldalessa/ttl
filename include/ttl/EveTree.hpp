#pragma once

#include "ExecutableTree.hpp"
#define EVE_FORCEINLINE
#include <eve/function/load.hpp>
#include <eve/function/store.hpp>
#include <eve/wide.hpp>

namespace ttl::eve
{
using namespace ::eve;

union Node {
  int offset = 0;
  double d;

  constexpr Node() {};
  constexpr Node(int i) : offset(i) {}
  constexpr Node(double d) : d(d) {}
};

template <int Depth, exe::Tag... ts>
struct Tree
{
  constexpr static int M = sizeof...(ts);
  constexpr static exe::Tag tags[M] = { ts... };
  int lhs_offset;
  Node data[M];

  template <int M>
  constexpr Tree(const ExecutableTree<M, Depth>& tree)
      : lhs_offset(tree.lhs_offset)
  {
    for (int i = 0; i < M; ++i) {
      if (tree.data[i].tag == exe::SCALAR) {
        std::construct_at(&data[i].offset, tree.data[i].offset);
      }
      if (tree.data[i].tag == exe::CONSTANT) {
        std::construct_at(&data[i].offset, tree.data[i].offset);
      }
      else if (tree.data[i].tag == exe::IMMEDIATE) {
        std::construct_at(&data[i].d, tree.data[i].d);
      }
    }
  }

  template <int j>
  [[gnu::always_inline]]
  void eval(int i, auto const& scalars, auto const& constants, auto& stack, int& d) const
  {
    if constexpr (tags[j] == exe::SUM) {
      stack[d - 2] += stack[d - 1];
      --d;
    }
    else if constexpr (tags[j] == exe::DIFFERENCE) {
      stack[d - 2] -= stack[d - 1];
      --d;
    }
    else if constexpr (tags[j] == exe::PRODUCT) {
      stack[d - 2] *= stack[d - 1];
      --d;
    }
    else if constexpr (tags[j] == exe::RATIO) {
      stack[d - 2] /= stack[d - 1];
      --d;
    }
    else if constexpr (tags[j] == exe::IMMEDIATE) {
      stack[d++] = data[j].d;
    }
    else if constexpr (tags[j] == exe::SCALAR) {
      auto* s = &scalars(data[j].offset, i);
      auto aligned = eve::as_aligned<alignof(wide<double>)>(s);
      stack[d++] = eve::load(aligned, eve::as_<eve::wide<double>>{});
    }
    else {
      static_assert(tags[j] == exe::CONSTANT);
      stack[d++] = constants(data[j].offset);
    }
  }

  [[gnu::always_inline]]
  eve::wide<double> eval(int i, auto const& scalars, auto const& constants) const
  {
    eve::wide<double> stack[Depth];
    int d = 0;

    [&]<std::size_t... j>(std::index_sequence<j...>) {
      (eval<j>(i, scalars, constants, stack, d), ...);
    }(std::make_index_sequence<M>());

    return stack[0];
  }

  [[gnu::always_inline]]
  // [[gnu::noinline]]
  void evaluate(int n, auto&& lhs, auto&& scalars, auto&& constants) const
  {
    constexpr static int N = eve::wide<double>::static_size;
    for (int i = 0; i < n; i += N)
    {
      auto* s = &lhs(lhs_offset, i);
      auto aligned = eve::as_aligned<alignof(wide<double>)>(s);
      eve::store(eval(i, scalars, constants), aligned);
    }
  }
};
}
