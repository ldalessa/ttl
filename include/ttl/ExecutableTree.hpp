#pragma once

#include "Rational.hpp"
#include "Tag.hpp"
#include "ScalarTree.hpp"
#include "Tensor.hpp"
#include "utils.hpp"
#include <fmt/core.h>
#include <memory>

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

  constexpr ExecutableTreeNode() {}

  constexpr const ExecutableTreeNode* a() const {
    return this - left;
  }

  constexpr const ExecutableTreeNode* b() const {
    return this - 1;
  }

  std::string to_string() const {
    switch (tag) {
     case SUM:
     case DIFFERENCE:
     case PRODUCT:
     case RATIO:
      return fmt::format("({} {} {})", a()->to_string(), tag, b()->to_string());

     case RATIONAL: return fmt::format("{}", q);
     case DOUBLE:   return fmt::format("{}", d);
     case TENSOR:
      if (constant) {
        return fmt::format("c({})", offset);
      }
      else {
        return fmt::format("s({})", offset);
      }
     default: assert(false);
    }
  }
};

template <int M>
struct ExecutableTree
{
  ExecutableTreeNode data[M];

  constexpr ExecutableTree(const ScalarTree* tree, auto const& scalars, auto const& constants)
  {
    assert(M == tree->size());
    auto i = map(M - 1, tree, scalars, constants);
    assert(i == M);
  }

  constexpr int size() const {
    return M;
  }

  constexpr const ExecutableTreeNode* root() const {
    return data + M - 1;
  }

  std::string to_string() const {
    return data[M - 1].to_string();
  }

 private:

  constexpr int map(int i, const ScalarTree* tree, auto const& scalars, auto const& constants)
  {
    switch (tree->tag)
    {
     case SUM:
     case DIFFERENCE:
     case PRODUCT:
     case RATIO:  return map_binary(i, tree, scalars, constants);
     case TENSOR: return map_tensor(i, tree, scalars, constants);
     case RATIONAL: return map_rational(i, tree);
     case DOUBLE: return map_double(i, tree);
     default: assert(false);
    }
  }

  constexpr int map_binary(int i, const ScalarTree* tree, auto const& scalars, auto const& constants)
  {
    int b = map(i - 1, tree->b(), scalars, constants);
    int a = map(i - (b + 1), tree->a(), scalars, constants);
    data[i].tag = tree->tag;
    data[i].left = b + 1;
    return a + b + 1;
  }

  constexpr int map_tensor(int i, const ScalarTree* tree, auto const& scalars, auto const& constants)
  {
    data[i].tag = TENSOR;
    data[i].constant = tree->constant;
    if (tree->constant) {
      data[i].offset = constants.find(tree);
    }
    else {
      data[i].offset = scalars.find(tree);
    }
    return 1;
  }

  constexpr int map_rational(int i, const ScalarTree* tree)
  {
    data[i].tag = RATIONAL;
    std::construct_at(&data[i].q, tree->q); // to change active member
    return 1;
  }

  constexpr int map_double(int i, const ScalarTree* tree)
  {
    data[i].tag = DOUBLE;
    std::construct_at(&data[i].d, tree->d); // to change active member
    return 1;
  }
};
}
