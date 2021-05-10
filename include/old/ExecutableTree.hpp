#pragma once

#include "ttl/exec.hpp"
#include "ttl/SerializedTree.hpp"

namespace ttl
{
  /// An executable tree.
  ///
  /// We're not using constexpr executable stack addresses during evaluation
  /// because we want to support multithreaded tree evaluation (so each thread
  /// has its own stack, and we don't know its address statically). We could
  /// create a stack array of some max size, template on the thread-id, and then
  /// these addresses _would_ be constexpr.
  template <class T, TreeShape shape, serialized_tree auto tree>
  struct ExecutableTree
  {
    using Stack = T[shape.stack_depth];
    constexpr static int N = shape.dims;

    template <int k>
    void eval_sum(Stack& stack) const
    {
      // c = a + b
      constexpr static int l = tree.left(k);
      constexpr static int r = tree.right(k);

      constexpr static int rk = tree.stack_offset(k);
      constexpr static int rl = tree.stack_offset(l);
      constexpr static int rr = tree.stack_offset(r);

      // Not constexpr (see class note on multithreading).
      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      constexpr static exec::Index ci = tree.index(k);
      constexpr static exec::Index ai = tree.index(l);
      constexpr static exec::Index bi = tree.index(r);

      static_assert(ci == ai);

      // Map the index space.
      constexpr static int M = ci.size();
      constexpr static std::array b_map = exec::make_map<N, M>(ci, bi);

      for (int i = 0; i < b_map.size(); ++i) {
        c[i] = a[i] + b[b_map[i]];
      }
    }

    template <int k>
    void eval_difference(Stack& stack) const
    {
      // c = a - b
      constexpr static int l = tree.left(k);
      constexpr static int r = tree.right(k);

      constexpr static int rk = tree.stack_offset(k);
      constexpr static int rl = tree.stack_offset(l);
      constexpr static int rr = tree.stack_offset(r);

      // Not constexpr (see class note on multithreading).
      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      constexpr static exec::Index ci = tree.index(k);
      constexpr static exec::Index ai = tree.index(l);
      constexpr static exec::Index bi = tree.index(r);

      static_assert(ci == ai);

      // Map the index space.
      constexpr static int M = ci.size();
      constexpr static std::array b_map = exec::make_map<N, M>(ci, bi);

      for (int i = 0; i < b_map.size(); ++i) {
        c[i] = a[i] - b[b_map[i]];
      }
    }

    template <int k>
    void eval_product(Stack& stack) const
    {
      // c = a * b
      constexpr static int l = tree.left(k);
      constexpr static int r = tree.right(k);

      constexpr static int rk = tree.stack_offset(k);
      constexpr static int rl = tree.stack_offset(l);
      constexpr static int rr = tree.stack_offset(r);

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      constexpr static exec::Index  ci = tree.index(k);
      constexpr static exec::Index all = tree.inner_index(k);
      constexpr static exec::Index  ai = tree.index(l);
      constexpr static exec::Index  bi = tree.index(r);

      // Map the index space.
      constexpr static int M = all.size();
      constexpr static std::array c_map = exec::make_map<N, M>(all, ci);
      constexpr static std::array a_map = exec::make_map<N, M>(all, ai);
      constexpr static std::array b_map = exec::make_map<N, M>(all, bi);

      // Don't know the state of the stack but we're going to need to accumulate
      // there so we need to zero it first (it's nearly certainly dirty, either
      // from previous frame or from previous evaluation)
      for (int i = 0; i < c_map.size(); ++i) {
        c[i] = T();
      }

      for (int i = 0; i < c_map.size(); ++i) {
        c[c_map[i]] += a[a_map[i]] * b[b_map[i]];
      }
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

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      auto rb = T(1)/b[0];
      for (int i = 0; i < ttl::pow(N, ci.size()); ++i) {
        c[i] = a[i] * rb;
      }
    }

    template <int k>
    void eval_immediate(Stack& stack) const
    {
      constexpr static double immediate = tree.immediate(k);
      stack[tree.stack_offset(k)] = immediate;
    }

    template <int k>
    void eval_scalar(int i, Stack& stack, auto const& scalars) const
    {
      constexpr static exec::Index  outer_index = tree.index(k);
      constexpr static exec::Index    all_index = tree.inner_index(k);
      constexpr static exec::Index tensor_index = tree.tensor_index(k);

      constexpr static int const *ids = tree.scalar_ids(k);
      constexpr static int rk = tree.stack_offset(k);

      // not constexpr addresses, see class note about multithreading
      T* const __restrict c = stack + rk;

      // Don't know the state of the stack but we're going to need to accumulate
      // there so we need to zero it first (it's nearly certainly dirty, either
      // from previous frame or from previous evaluation).
      for (int ii = 0; ii < ttl::pow(N, outer_index.size()); ++ii) {
        c[ii] = T();
      }

      constexpr static int M = all_index.size();
      constexpr static std::array c_map = exec::make_map<N, M>(all_index, outer_index);
      constexpr static std::array id_map = exec::make_map<N, M>(all_index, tensor_index);

      for (int ii = 0; ii < c_map.size(); ++ii) {
        c[c_map[ii]] += scalars(ids[id_map[ii]], i);
      }
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
