#pragma once

#include "Rational.hpp"
#include "Tag.hpp"
#include "ScalarTree.hpp"
#include "Tensor.hpp"
#include "utils.hpp"

namespace ttl {
struct ExecutableTreeNode
{
  Tag tag = TENSOR;
  int left = 0;
  bool constant = true;
  union
  {
    int offset = 0;
    Rational q;
    double   d;
  };

  constexpr const ExecutableTreeNode* a() const {
    return this - left;
  }

  constexpr const ExecutableTreeNode* b() const {
    return this - 1;
  }
};

template <int M>
struct ExecutableTree
{
  ExecutableTreeNode data[M];

  constexpr ExecutableTree(int N, const ScalarTree* tree, auto&& partials)
  {
    int i = M;
    auto op = [&](const ScalarTree* tree, auto&& self) -> int {
      auto n = --i;
      switch (tree->tag)
      {
       case SUM:
       case DIFFERENCE:
       case PRODUCT:
       case RATIO:  {
         int a = self(tree->b(), self);
         int b = self(tree->a(), self);
         assert(a == n - 1);
         data[n].tag = tree->tag;
         data[n].left = b;
         return n;
       }

       case TENSOR: {
         data[n].tag = TENSOR;
         data[n].constant = tree->constant;
         data[n].offset = partials.find(N, tree);
         return n;
       }

       case RATIONAL: {
         data[n].tag = RATIONAL;
         data[n].q = tree->q;
         return n;
       }

       case DOUBLE:  {
         data[n].tag = DOUBLE;
         data[n].d = tree->d;
         return n;
       }

       default: assert(false);
      }
    };
    op(tree, op);
  }
};
}
