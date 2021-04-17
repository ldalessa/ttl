#pragma once

#include "ttl/Scalar.hpp"
#include "ttl/Tag.hpp"
#include "ttl/TensorTree.hpp"
#include "ttl/set.hpp"

namespace ttl
{
  template <class T, int N, int Nodes, int Indices, int Scalars, int Immediates>
  struct ExecutableTensorTree
  {
    using Node = TensorTree::Node;

    char    indices[Indices]{};
    int  scalar_ids[Scalars]{};
    T    immediates[Immediates]{};

    Tag      tags[Nodes]{};
    int       rvo[Nodes]{};
    int     index[Nodes]{};
    int    scalar[Nodes]{};
    int immediate[Nodes]{};
    int      left[Nodes]{};

    struct cursor {
      int index = 0;
      int scalar = 0;
      int immediate = 0;
      int stack = 0;
    };

    constexpr ExecutableTensorTree(TensorTree const& tree, set<Scalar> const& scalars, set<Scalar> const& constants)
    {
      cursor c;
      map_(Nodes - 1, tree.root(), c, scalars, constants);
      assert(c.scalar == Scalars);
      // assert(c.index == Indices);
      assert(c.immediate == Immediates);
    }

    /// Record the information associated with a leaf node
    constexpr void record_leaf_(int i, Node const* tree, cursor& c)
    {
      tags[i]      = tree->tag;
      rvo[i]       = c.stack;
      index[i]     = c.index;
      scalar[i]    = c.scalar;
      immediate[i] = c.immediate;
      left[i]      = i;

      c.stack += tree->tensor_size(N);
    }

    /// Record the information associated with an index
    constexpr void record_index_(Node const* tree, cursor& c)
    {
      // Store my index to the right offset.
      for (char i : tree->index) {
        indices[c.index++] = i;
      }
    }

    constexpr int map_(int i, Node const* tree, cursor& c, set<Scalar> const& scalars, set<Scalar> const& constants)
    {
      switch (tree->tag)
      {
       case SUM:
       case DIFFERENCE:
       case PRODUCT:
       case RATIO:     return map_binary_(i, tree, c, scalars, constants);
       case INDEX:     return map_index_(i, tree, c);
       case TENSOR:    return map_tensor_(i, tree, c, scalars, constants);
       case RATIONAL:  return map_rational_(i, tree, c);
       case DOUBLE:    return map_double_(i, tree, c);
       default: assert(false);
      }
      __builtin_unreachable();
    }

    constexpr int map_binary_(int i, Node const* tree, cursor& c, set<Scalar> const& scalars, set<Scalar> const& constants)
    {
      int b = map_(i - 1, tree->b(), c, scalars, constants);
      int a = map_(i - (b + 1), tree->a(), c, scalars, constants);
      record_index_(tree, c);
      switch (tree->tag) {
       case SUM:        tags[i] = SUM; break;
       case DIFFERENCE: tags[i] = DIFFERENCE; break;
       case PRODUCT:    tags[i] = PRODUCT; break;
       case RATIO:      tags[i] = RATIO; break;
       default: assert(false);
      }
      return a + b + 1;
    }

    constexpr int map_tensor_(int i, Node const* tree, cursor& c, set<Scalar> const& scalars, set<Scalar> const& constants)
    {
      record_leaf_(i, tree, c);
      record_index_(tree, c);

      // Store my scalar offsets to the scalar_ids array.
      tree->scalars(N, [&](Scalar const& s) {
        if (s.constant) {
          auto i = constants.find(s);
          assert(i);
          scalar_ids[c.scalar++] = *i;
        }
        else {
          auto i = scalars.find(s);
          assert(i);
          scalar_ids[c.scalar++] = *i;
        }
      });

      return 1;
    }

    constexpr int map_rational_(int i, Node const* tree, cursor& c)
    {
      record_leaf_(i, tree, c);
      immediates[c.immediate++] = as<T>(tree->q);
      return 1;
    }

    constexpr int map_double_(int i, Node const* tree, cursor& c)
    {
      record_leaf_(i, tree, c);
      immediates[c.immediate++] = tree->d;
      return 1;
    }

    constexpr int map_index_(int i, Node const* tree, cursor& c)
    {
      assert(tree->index.size() == 2);
      record_leaf_(i, tree, c);
      record_index_(tree, c);

      // this is stupid
      ScalarIndex index(2);
      do {
        immediates[c.immediate++] = (index[0] == index[1]) ? 1 : 0;
      } while (index.carry_sum_inc(N));
      return 1;
    }
  };
}
