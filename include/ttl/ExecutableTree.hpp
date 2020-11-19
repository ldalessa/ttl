#pragma once

#include "Rational.hpp"
#include "Tag.hpp"
#include "ScalarTree.hpp"
#include <fmt/core.h>

namespace ttl
{
/// The executable tree represents the structure that we actually evaluate at
/// runtime in order compute the right hand side for a single scalar function.
template <int M, int Depth>
struct ExecutableTree
{
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
    Tag    tag;
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

  int lhs_offset;
  Node data[M];

  constexpr ExecutableTree(const ScalarTree& tree, auto const& scalars, auto const& constants)
      : lhs_offset(scalars.find(tree.lhs()))
  {
    auto i = map(M - 1, tree.root(), scalars, constants);
    assert(i == M);
  }

  [[gnu::always_inline]]
  double eval(int i, auto const& scalars, auto const& constants) const
  {
    double stack[Depth];
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
       case SUM:        stack[d - 2] += stack[d - 1]; --d; break;
       case DIFFERENCE: stack[d - 2] -= stack[d - 1]; --d; break;
       case PRODUCT:    stack[d - 2] *= stack[d - 1]; --d; break;
       case RATIO:      stack[d - 2] /= stack[d - 1]; --d; break;
       case IMMEDIATE:  stack[d++] = data[j].d; break;
       case SCALAR:     stack[d++] = scalars(data[j].offset, i); break;
       case CONSTANT:   stack[d++] = constants(data[j].offset); break;
      }
    }

    return stack[0];
  }

  [[gnu::always_inline]]
  void evaluate(int n, auto const& lhs, auto const& scalars, auto const& constants) const
  {
    for (int i = 0; i < n; ++i)
    {
      lhs(lhs_offset, i) = eval(i, scalars, constants);
    }
  }

  std::string to_string() const {
    std::string stack[Depth];
    int n = 0;
    for (auto&& node : data) {
      if (node.tag < IMMEDIATE) {
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
     case ttl::SUM:        data[i].tag = SUM; break;
     case ttl::DIFFERENCE: data[i].tag = DIFFERENCE; break;
     case ttl::PRODUCT:    data[i].tag = PRODUCT; break;
     case ttl::RATIO:      data[i].tag = RATIO; break;
     default: assert(false);
    }
    return a + b + 1;
  }

  constexpr int map_tensor(int i, const ScalarTree::Node* tree, auto const& scalars, auto const& constants)
  {
    if (tree->constant) {
      data[i].tag = CONSTANT;
      data[i].offset = constants.find(tree);
    }
    else {
      data[i].tag = SCALAR;
      data[i].offset = scalars.find(tree);
    }
    return 1;
  }

  constexpr int map_rational(int i, const ScalarTree::Node* tree)
  {
    data[i].tag = IMMEDIATE;
    data[i].d = to_double(tree->q);
    return 1;
  }

  constexpr int map_double(int i, const ScalarTree::Node* tree)
  {
    data[i].tag = IMMEDIATE;
    data[i].d = tree->d;
    return 1;
  }
};
}
