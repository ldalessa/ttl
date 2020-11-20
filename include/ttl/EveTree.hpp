#pragma once

#include "ExecutableTree.hpp"
#include <eve/function/load.hpp>
#include <eve/function/store.hpp>
#include <eve/wide.hpp>

namespace ttl
{
/// Executable tree that encodes tree tag geometry in its type.
///
/// This Tree allows us to implement the kernel stack machine, i.e., the
/// evaluation of the tree on runtime data, through an `if constexpr` dispatch
/// tree, rather than a runtime switch operation.
///
/// The benefit to this structure is that it does not rely on the compiler's
/// inlining and loop unrolling and constant propagation in order to eliminate
/// the runtime switch operation, because the switch operation does not
/// exist. This means that we can use SIMD vector data (in the form of eve wide
/// vectors) when processing the vector data without worrying about the compiler
/// "giving up" on optimizing the tree.
///
/// @tparam   Depth The tree depth is used to allocate stack space for running
///                 the kernel's stack machine.
/// @tparam   ts... The RPN tree tag types.
template <int Depth, exe::Tag... ts>
struct EveTree
{
  // Capture the tags into a constexpr array (they're hard to use otherwise).
  constexpr static int M = sizeof...(ts);
  constexpr static exe::Tag tags[M] = { ts... };

  // The scalar field offset for the left-hand-side of this expression. For
  // instance, this might be the offset of the x component of velocity.
  int lhs_offset;

  // The tree node data. At most one of these is active for any `i`, but it
  // doesn't impact compile time right now and using a union is a little more
  // complex. Could also use an array of exe::Node, but it would store redundant
  // tags in that case so I chose not to (I realize this is inconsistent with
  // the rationale for not using a union).
  int       offset[M] = {};
  double immediate[M] = {};

  /// Construct an eve tree from an executable tree.
  template <int N>
  constexpr EveTree(const ExecutableTree<N, Depth>& tree)
      : lhs_offset(tree.lhs_offset)
  {
    assert(N == M);
    for (int i = 0; i < M; ++i) {
      assert(tags[i] == tree.data[i].tag);
      if (tags[i] == exe::SCALAR) {
        offset[i] = tree.data[i].offset;
      }
      else if (tags[i] == exe::CONSTANT) {
        offset[i] = tree.data[i].offset;
      }
      else if (tags[i] == exe::IMMEDIATE) {
        immediate[i] = tree.data[i].d;
      }
    }
  }

  /// Evaluate one step for the kernel.
  ///
  /// This leverages the fact that the tree tag geometry is known as constexpr
  /// here, allowing us to use the `if constexpr` switch structure and forcing
  /// the compiler into "seeing through" the switch. Externally this is
  /// evaluated as the function in a fold operation across the whole
  /// expression.
  ///
  /// The reference to the external stack defines the type of operation we're
  /// dealing with, for SIMD evaluation we'll have a stack of SIMD vectors, and
  /// for scalar evaluation we'll have a stack of doubles.
  ///
  /// @tparam            j Index of the node to evaluate.
  ///
  /// @param             i The SIMD index into the outer struct-of-array.
  /// @param       scalars The struct-of-array scalar storage functor.
  /// @param     constants The functor map for non-immediate tree constants.
  /// @param[in/out] stack The kernel's stack machine storage.
  /// @param[in/out]     d The current top of the stack.
  template <int j>
  [[gnu::always_inline]]
  void eval_kernel_step(int i, auto const& scalars, auto const& constants, auto& stack, int& d) const
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
      stack[d++] = immediate[j];
    }
    else if constexpr (tags[j] == exe::SCALAR) {
      load(stack[d++], &scalars(offset[j], i));
    }
    else {
      static_assert(tags[j] == exe::CONSTANT);
      stack[d++] = constants(offset[j]);
    }
  }

  /// Evaluate the stack machine for the kernel.
  ///
  /// This expands the stack machine via an index-sequence and fold operation,
  /// which forces the compiler to inline the entire machine and thus provides
  /// visibility to the entire kernel at once. This is _required_ to get good
  /// performance out of the kernel.
  ///
  /// It takes a reference to the external stack for the machine, which allows
  /// the same kernel code to be reused for both scalar and vector evaluation.
  ///
  /// @param             i The SIMD index into the outer struct-of-array.
  /// @param       scalars The struct-of-array scalar storage functor.
  /// @param     constants The functor map for non-immediate tree constants.
  /// @param[in/out] stack The kernel's stack machine storage.
  ///
  /// @returns The result of executing the kernel on index i.
  [[gnu::always_inline]]
  auto eval_kernel(int i, auto const& scalars, auto const& constants, auto& stack) const
  {
    int d = 0;

    [&]<std::size_t... j>(std::index_sequence<j...>) {
      (eval_kernel_step<j>(i, scalars, constants, stack, d), ...);
    }(std::make_index_sequence<M>());

    return stack[--d];
  }

  /// Evaluate the kernel in wide vector form.
  ///
  /// This allocates a stack of SIMD vectors and forwards to the kernel.
  ///
  /// @param             i The SIMD index into the outer struct-of-array.
  /// @param       scalars The struct-of-array scalar storage functor.
  /// @param     constants The functor map for non-immediate tree constants.
  ///
  /// @returns The wide result of executing the kernel on index i.
  [[gnu::always_inline]]
  eve::wide<double> eval_wide(int i, auto const& scalars, auto const& constants) const
  {
    eve::wide<double> stack[Depth];
    return eval_kernel(i, scalars, constants, stack);
  }

  /// Evaluate the kernel in scalar form.
  ///
  /// This allocates a stack of scalars and forwards to the kernel.
  ///
  /// @param             i The SIMD index into the outer struct-of-array.
  /// @param       scalars The struct-of-array scalar storage functor.
  /// @param     constants The functor map for non-immediate tree constants.
  ///
  /// @returns The scalar result of executing the kernel on index i.
  [[gnu::noinline]]
  double eval_scalar(int i, auto const& scalars, auto const& constants) const
  {
    double stack[Depth];
    return eval_kernel(i, scalars, constants, stack);
  }

  /// Evaluate the kernel for the entire
  ///
  /// This allocates a stack of scalars and forwards to the kernel.
  ///
  /// @param             n The length of the vector data.
  /// @param       scalars The struct-of-array scalar storage functor.
  /// @param     constants The functor map for non-immediate tree constants.
  [[gnu::always_inline]]
  void evaluate(int n, auto&& lhs, auto&& scalars, auto&& constants) const
  {
    constexpr int A = eve::wide<double>::static_alignment;
    constexpr int N = eve::wide<double>::static_size;

    // loop induction variable
    int i = 0;

    // process the potentially unaligned prefix of the vector
    std::intptr_t base = reinterpret_cast<std::intptr_t>(&lhs(lhs_offset, 0));
    for (int e = base % A; i < e; ++i) {
      store(&lhs(lhs_offset, i), eval_scalar(i, scalars, constants));
    }

    // process the body of the vector in eve::wide<double> chunks
    for (int e = n / N; i < e; i += N) {
      store(&lhs(lhs_offset, i), eval_wide(i, scalars, constants));
    }

    // process any remaining scalars
    for (; i < n; ++i) {
      store(&lhs(lhs_offset, i), eval_scalar(i, scalars, constants));
    }
  }

 private:
  /// Load a scalar into a stack slot (for the kernel's stack machine).
  static void load(double& slot, const double* addr) {
    slot = *addr;
  }

  /// Load a wide vector into a stack slot (for the kernel's stack machine).
  static void load(eve::wide<double>& slot, const double* addr) {
    constexpr int A = eve::wide<double>::static_alignment;
    auto aligned = eve::as_aligned<A>(addr);
    slot = eve::load(aligned, eve::as_<eve::wide<double>>{});
  }

  /// Store a scalar.
  static void store(double* addr, double value) {
    *addr = value;
  }

  /// Store a wide vector.
  static void store(double* addr, eve::wide<double>&& value) {
    constexpr int A = eve::wide<double>::static_alignment;
    auto aligned = eve::as_aligned<A>(addr);
    eve::store(std::move(value), aligned);
  }
};
}
