#pragma once

#include "ttl/exec.hpp"
#include "ttl/SerializedTree.hpp"
#include <array>

namespace ttl
{
  template <class T, TreeShape shape, auto tree>
  struct ExecutableTree
  {
    using Stack = T[shape.stack_depth];
    constexpr static int N = shape.dims;

    std::array<T, shape.n_immediates> immediates;

    constexpr ExecutableTree(std::array<T, shape.n_immediates> immediates)
        : immediates(std::move(immediates))
    {
    }

    template <int k>
    void eval_sum(Stack& stack) const
    {
      // c = a + b
      constexpr static int l = tree.left(k);
      constexpr static int r = tree.right(k);

      constexpr static exec::Index ci = tree.index(k);
      constexpr static exec::Index ai = tree.index(l);
      constexpr static exec::Index bi = tree.index(r);

      static_assert(ci == ai);

      constexpr static exec::IndexMapper<ci, bi> bmap{};

      constexpr static int M = ci.size();

      constexpr static int rk = tree.stack_offset(k);
      constexpr static int rl = tree.stack_offset(l);
      constexpr static int rr = tree.stack_offset(r);

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      exec::eval<N, M>([&](std::array<int, M> outer) {
        int i = exec::row_major<N>(outer);
        int j = exec::row_major<N>(bmap(outer));
        c[i] = a[i] + b[j];
      });
    }

    template <int k>
    void eval_difference(Stack& stack) const
    {
      // c = a - b
      constexpr static int l = tree.left(k);
      constexpr static int r = tree.right(k);

      constexpr static exec::Index ci = tree.index(k);
      constexpr static exec::Index ai = tree.index(l);
      constexpr static exec::Index bi = tree.index(r);
      static_assert(ci == ai);

      constexpr static exec::IndexMapper<ci, bi> bmap{};

      constexpr static int M = ci.size();

      constexpr static int rk = tree.stack_offset(k);
      constexpr static int rl = tree.stack_offset(l);
      constexpr static int rr = tree.stack_offset(r);

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      exec::eval<N, M>([&](std::array<int, M> outer) {
        int i = exec::row_major<N>(outer);
        int j = exec::row_major<N>(bmap(outer));
        c[i] = a[i] - b[j];
      });
    }

    template <int k>
    void eval_product(Stack& stack) const
    {
      // c = a * b
      constexpr static int l = tree.left(k);
      constexpr static int r = tree.right(k);

      constexpr static exec::Index  ci = tree.index(k);
      constexpr static exec::Index all = tree.inner_index(k);
      constexpr static exec::Index  ai = tree.index(l);
      constexpr static exec::Index  bi = tree.index(r);

      constexpr static exec::IndexMapper<all, ci> cmap{};
      constexpr static exec::IndexMapper<all, ai> amap{};
      constexpr static exec::IndexMapper<all, bi> bmap{};

      constexpr static int rk = tree.stack_offset(k);
      constexpr static int rl = tree.stack_offset(l);
      constexpr static int rr = tree.stack_offset(r);

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      // Don't know the state of the stack but we're going to need to accumulate
      // there so we need to zero it first (it's nearly certainly dirty, either
      // from previous frame or from previous evaluation)
      for (int ii = 0; ii < ttl::pow(N, ci.size()); ++ii) {
        c[ii] = T();
      }

      constexpr static int M = all.size();
      exec::eval<N, M>([&](std::array<int, M> outer)
      {
        int ii = exec::row_major<N>(cmap(outer));
        int jj = exec::row_major<N>(amap(outer));
        int kk = exec::row_major<N>(bmap(outer));
        c[ii] += a[jj] * b[kk];
      });
    }

    template <int k>
    void eval_ratio(Stack& stack) const
    {
      // c = a / b
      constexpr static int l = tree.left(k);
      constexpr static int r = tree.right(k);

      // right now we're expecting `b` to be a scalar.
      constexpr static exec::Index  ci = tree.index(k);
      constexpr static exec::Index all = tree.inner_index(k);
      constexpr static exec::Index  ai = tree.index(l);
      constexpr static exec::Index  bi = tree.index(r);

      static_assert(ci == ai);
      static_assert(ci == all);
      static_assert(bi.size() == 0);

      constexpr static int rk = tree.stack_offset(k);
      constexpr static int rl = tree.stack_offset(l);
      constexpr static int rr = tree.stack_offset(r);

      constexpr static int M = ttl::pow(N, ci.size());

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      auto rb = T(1)/b[0];
      for (int i = 0; i < M; ++i) {
        c[i] = a[i] * rb;
      }
    }

    template <int k>
    void eval_immediate(Stack& stack) const
    {
      stack[tree.stack_offset(k)] = immediates[tree.immediate_offsets[k]];
    }

    template <int k>
    void eval_scalar(int i, Stack& stack, auto const& scalars) const
    {
      constexpr static exec::Index outer = tree.index(k);
      constexpr static exec::Index   all = tree.inner_index(k);
      constexpr static exec::Index index = tree.tensor_index(k);

      constexpr static exec::IndexMapper<all, outer> outer_map{};
      constexpr static exec::IndexMapper<all, index> index_map{};

      constexpr static int rk = tree.stack_offset(k);

      constexpr static int const *ids = tree.scalar_ids(k);

      T* const __restrict c = stack + rk;

      // Don't know the state of the stack but we're going to need to accumulate
      // there so we need to zero it first (it's nearly certainly dirty, either
      // from previous frame or from previous evaluation).
      for (int ii = 0; ii < ttl::pow(N, outer.size()); ++ii) {
        c[ii] = T();
      }

      constexpr static int M = all.size();
      exec::eval<N, M>([&](std::array<int, M> outer)
      {
        int ii = exec::row_major<N>(outer_map(outer));
        int jj = exec::row_major<N>(index_map(outer));
        c[ii] += scalars(ids[jj], i);
      });
    }

    template <int k>
    void eval_constant(Stack& stack, auto const& constants) const
    {
      constexpr static int const *ids = tree.scalar_ids(k);
      constexpr static int const *end = tree.scalar_ids(k + 1);
      constexpr static int M = end - ids;

      static constexpr int rk = tree.stack_offset(k);

      T* __restrict c = stack + rk;

      for (int i = 0; i < M; ++i) {
        c[i] = constants(ids[i]);
      }
    }

    template <int k>
    void eval_delta(Stack& stack) const
    {
      static constexpr int rk = tree.stack_offset(k);
      T* __restrict c = stack + rk;

      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          c[i * N + j] = T(i == j);
        }
      }
    }

    template <int k>
    void eval_kernel_step(int i, Stack& stack, auto const& scalars, auto const& constants) const
    {
      if constexpr (tree.tags[k] == exec::SUM) {
        eval_sum<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::DIFFERENCE) {
        eval_difference<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::PRODUCT) {
        eval_product<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::RATIO) {
        eval_ratio<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::IMMEDIATE) {
        eval_immediate<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::CONSTANT) {
        eval_constant<k>(stack, constants);
      }
      if constexpr (tree.tags[k] == exec::SCALAR) {
        eval_scalar<k>(i, stack, scalars);
      }
      if constexpr (tree.tags[k] == exec::DELTA) {
        eval_delta<k>(stack);
      }
    }

    void evaluate(auto const& scalars, auto const& constants) const
    {
      Stack stack{};
      [&]<std::size_t... i>(std::index_sequence<i...>) {
        (eval_kernel_step<i>(0, stack, scalars, constants), ...);
      }(std::make_index_sequence<shape.n_nodes>());
    }
  };
}
