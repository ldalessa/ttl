#pragma once

#include "Rational.hpp"
#include "Tag.hpp"
#include "ScalarTree.hpp"
#define EVE_FORCEINLINE
#include <eve/function/load.hpp>
#include <eve/function/store.hpp>
#include <eve/wide.hpp>
#include <fmt/core.h>

namespace ttl
{
namespace exe {
enum Tag {
  SUM,
  DIFFERENCE,
  PRODUCT,
  RATIO,
  IMMEDIATE,
  SCALAR,
  CONSTANT
};

struct Node
{
  Tag tag;
  int offset = 0;
  double   d = 0.0;

  std::string to_string() const {
    switch (tag) {
     case SUM:        return "+";
     case DIFFERENCE: return "-";
     case PRODUCT:    return "*";
     case RATIO:      return "/";
     case IMMEDIATE:  return fmt::format("{}", d);
     case SCALAR:     return fmt::format("s{}", offset);
     case CONSTANT:   return fmt::format("c{}", offset);
     default: assert(false);
    }
    __builtin_unreachable();
  }
};
}

/// The executable tree represents the structure that we actually evaluate at
/// runtime in order compute the right hand side for a single scalar function.
template <int M, int Depth>
struct ExecutableTree
{
  int lhs_offset;
  exe::Node data[M];

  constexpr ExecutableTree(const ScalarTree& tree, auto const& scalars, auto const& constants)
      : lhs_offset(scalars.find(tree.lhs()))
  {
    assert(scalars[lhs_offset] == tree.lhs());
    auto i = map(M - 1, tree.root(), scalars, constants);
    assert(i == M);
  }

  constexpr static int size() {
    return M;
  }

  constexpr static int depth() {
    return Depth;
  }

  [[gnu::always_inline]]
  eve::wide<double> eval(int i, auto const& scalars, auto const& constants) const
  {
    eve::wide<double> stack[Depth];
    int d = 0;

#ifdef __clang__
#pragma unroll
#else
#pragma GCC unroll 65534
#endif
    for (int j = 0; j < M; ++j)
    {
      switch (data[j].tag)
      {
       case exe::SUM:        stack[d - 2] += stack[d - 1]; --d; break;
       case exe::DIFFERENCE: stack[d - 2] -= stack[d - 1]; --d; break;
       case exe::PRODUCT:    stack[d - 2] *= stack[d - 1]; --d; break;
       case exe::RATIO:      stack[d - 2] /= stack[d - 1]; --d; break;
       case exe::IMMEDIATE:  stack[d++] = data[j].d; break;
       case exe::SCALAR:     stack[d++] = eve::load(&scalars(data[j].offset, i), eve::as_<eve::wide<double>>{}); break;
       case exe::CONSTANT:   stack[d++] = constants(data[j].offset); break;
      }
    }

    return stack[0];
  }

  // [[gnu::always_inline]]
  [[gnu::noinline]]
  void evaluate(int n, auto&& lhs, auto&& scalars, auto&& constants) const
  {
    constexpr int N = eve::wide<double>::static_size;
    for (int i = 0; i < n; i += N)
    {
      eve::store(eval(i, scalars, constants), &lhs(lhs_offset, i));
    }
  }

  std::string to_string() const {
    std::string stack[Depth];
    int n = 0;
    for (auto&& node : data) {
      if (node.tag < exe::IMMEDIATE) {
        std::string b = stack[--n];
        std::string a = stack[--n];
        stack[n++] = fmt::format("({} {} {})", a, node.to_string(), b);
      }
      else {
        stack[n++] = node.to_string();
      }
    }
    assert(n == 1);
    std::string s("s");
    std::string eq(" = ");
    return s += std::to_string(lhs_offset) += eq += stack[--n];
  }

 private:
  constexpr int map(int i, const ScalarTree::Node* tree, auto const& scalars, auto const& constants)
  {
    switch (tree->tag)
    {
     case ttl::SUM:
     case ttl::DIFFERENCE:
     case ttl::PRODUCT:
     case ttl::RATIO:     return map_binary(i, tree, scalars, constants);
     case ttl::TENSOR:    return map_tensor(i, tree, scalars, constants);
     case ttl::RATIONAL:  return map_rational(i, tree);
     case ttl::DOUBLE:    return map_double(i, tree);
     default: assert(false);
    }
    __builtin_unreachable();
  }

  constexpr int map_binary(int i, const ScalarTree::Node* tree, auto const& scalars, auto const& constants)
  {
    int b = map(i - 1, tree->b(), scalars, constants);
    int a = map(i - (b + 1), tree->a(), scalars, constants);
    switch (tree->tag) {
     case ttl::SUM:        data[i].tag = exe::SUM; break;
     case ttl::DIFFERENCE: data[i].tag = exe::DIFFERENCE; break;
     case ttl::PRODUCT:    data[i].tag = exe::PRODUCT; break;
     case ttl::RATIO:      data[i].tag = exe::RATIO; break;
     default: assert(false);
    }
    return a + b + 1;
  }

  constexpr int map_tensor(int i, const ScalarTree::Node* tree, auto const& scalars, auto const& constants)
  {
    if (tree->constant) {
      data[i].tag = exe::CONSTANT;
      data[i].offset = constants.find(tree);
    }
    else {
      data[i].tag = exe::SCALAR;
      data[i].offset = scalars.find(tree);
    }
    return 1;
  }

  constexpr int map_rational(int i, const ScalarTree::Node* tree)
  {
    data[i].tag = exe::IMMEDIATE;
    data[i].d = to_double(tree->q);
    return 1;
  }

  constexpr int map_double(int i, const ScalarTree::Node* tree)
  {
    data[i].tag = exe::IMMEDIATE;
    data[i].d = tree->d;
    return 1;
  }
};
}
