#pragma once

#include "Rational.hpp"
#include "Tag.hpp"
#include "ScalarTree.hpp"
#include "Tensor.hpp"
#include "utils.hpp"

namespace ttl {
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
       case IMMEDIATE:  return std::to_string(d);
       case SCALAR:     return std::string("s") += std::to_string(offset) += std::string("");
       case CONSTANT:   return std::string("c") += std::to_string(offset) += std::string("");
       default: assert(false);
      }
    }
  };

  Node data[M];

  constexpr ExecutableTree(const ScalarTree* tree, auto const& scalars, auto const& constants)
  {
    auto i = map(M - 1, tree, scalars, constants);
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
       case IMMEDIATE:  stack[d++] = data[j].immediate; break;
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
      lhs(i) = eval(i, scalars, constants);
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
    return stack[--n];
  }

 private:
  constexpr int map(int i, const ScalarTree* tree, auto const& scalars, auto const& constants)
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
  }

  constexpr int map_binary(int i, const ScalarTree* tree, auto const& scalars, auto const& constants)
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

  constexpr int map_tensor(int i, const ScalarTree* tree, auto const& scalars, auto const& constants)
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

  constexpr int map_rational(int i, const ScalarTree* tree)
  {
    data[i].tag = IMMEDIATE;
    data[i].d = to_double(tree->q);
    return 1;
  }

  constexpr int map_double(int i, const ScalarTree* tree)
  {
    data[i].tag = IMMEDIATE;
    data[i].d = tree->d;
    return 1;
  }
};
}
