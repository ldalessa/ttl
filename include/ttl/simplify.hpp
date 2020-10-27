#pragma once

#include "TaggedTree.hpp"

namespace ttl {
constexpr auto simplify(TaggedTree<INDEX> tree) {
  return tree;
}

constexpr auto simplify(TaggedTree<TENSOR> tree) {
  return tree;
}

constexpr auto simplify(TaggedTree<RATIONAL> tree) {
  return tree;
}

constexpr auto simplify(TaggedTree<DOUBLE> tree) {
  return tree;
}

}
