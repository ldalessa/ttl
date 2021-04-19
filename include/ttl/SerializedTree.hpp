#pragma once

#include "ttl/Scalar.hpp"
#include "ttl/Tag.hpp"
#include "ttl/TensorTree.hpp"
#include "ttl/TreeShape.hpp"
#include "ttl/exec.hpp"
#include "ttl/set.hpp"
#include <array>

namespace ttl
{
  template <class T, TreeShape shape>
  struct SerializedTree
  {
    using Node = TensorTree::Node;

    // for debugging
    decltype(shape) debug_shape = shape;

    // Arrays of data.
    char        indices_[shape.n_indices];
    char  inner_indices_[shape.n_inner_indices];
    char tensor_indices_[shape.n_tensor_indices];
    int      scalar_ids_[shape.n_scalars];
    // T        immediates[shape.n_immediates];

    exec::Tag tags [shape.n_nodes];
    int        rvo_[shape.n_nodes];
    int       left_[shape.n_nodes];

    int        index_offsets_[shape.n_nodes + 1];
    int  inner_index_offsets_[shape.n_nodes + 1];
    int tensor_index_offsets_[shape.n_nodes + 1];
    int   scalar_ids_offsets_[shape.n_nodes + 1];
    int    immediate_offsets [shape.n_nodes + 1];

    constexpr SerializedTree(TensorTree const& tree,
                             std::array<T, shape.n_immediates>& immediates,
                             set<Scalar> const& scalars,
                             set<Scalar> const& constants)
    {
      {
        Builder_ builder(*this, immediates);
        builder.map(tree.root(), scalars, constants);
      }

      // Just some extra checks for the tree integrity... don't really think any
      // of these should fail and they're redundant with other checks in the
      // builder, but whatever.
      for (int i = 0; i < shape.n_nodes; ++i) {
        assert(index_offsets_[i]        <= index_offsets_[i+1]);
        assert(inner_index_offsets_[i]  <= inner_index_offsets_[i+1]);
        assert(tensor_index_offsets_[i] <= tensor_index_offsets_[i+1]);
        assert(scalar_ids_offsets_[i]   <= scalar_ids_offsets_[i+1]);
        assert(immediate_offsets[i]     <= immediate_offsets[i+1]);

        if (is_binary(tags[i])) {
          assert(left_[i] < i - 1);
          assert(0 <= rvo_[i]);
          assert(rvo_[i] < rvo_[left_[i]]);
          assert(rvo_[left_[i]] < rvo_[i - 1]);
          assert(rvo_[i - 1] <= shape.stack_depth);
        }
      }
    }

    constexpr int left(int k) const
    {
      return left_[k];
    }

    constexpr int right(int k) const
    {
      return k - 1;
    }

    constexpr exec::Index index(int k) const
    {
      return exec::Index {
        .i = &indices_[index_offsets_[k]],
        .e = &indices_[index_offsets_[k+1]]
        };
    }

    constexpr exec::Index inner_index(int k) const
    {
      return exec::Index {
        .i = &inner_indices_[inner_index_offsets_[k]],
        .e = &inner_indices_[inner_index_offsets_[k+1]]
        };
    }

    constexpr exec::Index tensor_index(int k) const
    {
      return exec::Index {
        .i = &tensor_indices_[tensor_index_offsets_[k]],
        .e = &tensor_indices_[tensor_index_offsets_[k+1]]
        };
    }

    constexpr int stack_offset(int k) const
    {
      return rvo_[k];
    }

    constexpr int const * scalar_ids(int k) const
    {
      return &scalar_ids_[scalar_ids_offsets_[k]];
    }

    // Variables used during the initialization process.
    struct Builder_
    {
      SerializedTree& tree;
      std::array<T, shape.n_immediates>& immediates;

      int i = 0;
      int index = 0;
      int inner_index = 0;
      int tensor_index = 0;
      int scalar = 0;
      int immediate = 0;
      ce::dvector<int> stack;

      constexpr Builder_(SerializedTree& tree, std::array<T, shape.n_immediates>& immediates)
          : tree(tree)
          , immediates(immediates)
      {
        stack.reserve(shape.stack_depth + 1);
        stack.push_back(0);
      }

      constexpr ~Builder_()
      {
        // The maps are all size+1, so that we can compute the size of each
        // extent for the root. This is standard compressed-sparse-row design.
        tree.index_offsets_[i]        = std::size(tree.indices_);
        tree.inner_index_offsets_[i]  = std::size(tree.inner_indices_);
        tree.tensor_index_offsets_[i] = std::size(tree.tensor_indices_);
        tree.scalar_ids_offsets_[i]   = std::size(tree.scalar_ids_);
        tree.immediate_offsets[i]     = shape.n_immediates; // std::size(tree.immediates);

        assert(i == shape.n_nodes);
        assert(scalar == shape.n_scalars);
        assert(index == shape.n_indices);
        assert(inner_index == shape.n_inner_indices);
        assert(tensor_index == shape.n_tensor_indices);
        assert(immediate == shape.n_immediates);
        assert(stack.size() == 2);
      }

      constexpr exec::Tag to_tag(Node const* node)
      {
        switch (node->tag)
        {
         case ttl::SUM:        return exec::SUM;
         case ttl::DIFFERENCE: return exec::DIFFERENCE;
         case ttl::PRODUCT:    return exec::PRODUCT;
         case ttl::RATIO:      return exec::RATIO;
         case ttl::INDEX:      return exec::DELTA;
         case ttl::TENSOR:     return (node->constant) ? exec::CONSTANT : exec::SCALAR;
         case ttl::RATIONAL:   return exec::IMMEDIATE;
         case ttl::DOUBLE:     return exec::IMMEDIATE;
         default:
          assert(false);
        }
        __builtin_unreachable();
      }

      /// Record the information associated with a leaf node
      constexpr void record(Node const* node, int top_of_stack, int left = -1)
      {
        assert(top_of_stack + node->tensor_size(shape.dims) <= shape.stack_depth);
        tree.tags[i]                 = to_tag(node);
        tree.rvo_[i]                 = top_of_stack;
        tree.left_[i]                = left;

        tree.index_offsets_[i]        = index;
        tree.inner_index_offsets_[i]  = inner_index;
        tree.tensor_index_offsets_[i] = tensor_index;
        tree.scalar_ids_offsets_[i]   = scalar;
        tree.immediate_offsets[i]     = immediate;

        // Store my outer index to the right offset.
        for (char c : node->outer()) {
          tree.indices_[index++] = c;
        }

        // Store the all index to the right offset.
        for (char c : node->all()) {
          tree.inner_indices_[inner_index++] = c;
        }
      }

      constexpr void map_tensor(Node const* node, set<Scalar> const& scalars, set<Scalar> const& constants)
      {
        // Store my scalar offsets to the scalar_ids array.
        node->scalars(shape.dims, [&](Scalar const& s) {
            if (s.constant) {
              auto i = constants.find(s);
              assert(i);
              tree.scalar_ids_[scalar++] = *i;
            }
            else {
              auto i = scalars.find(s);
              assert(i);
              tree.scalar_ids_[scalar++] = *i;
            }
          });

        // Store my tensor index
        for (char c : node->index) {
          tree.tensor_indices_[tensor_index++] = c;
        }
      }

      constexpr int map(Node const* node, set<Scalar> const& scalars, set<Scalar> const& constants)
      {
        int top_of_stack = stack.back();
        stack.push_back(top_of_stack + node->tensor_size(shape.dims));

        if (!std::is_constant_evaluated()) {
          fmt::print("serialize stack:{}\n", stack.back());
        }

        switch (node->tag)
        {
         case ttl::SUM:
         case ttl::DIFFERENCE:
         case ttl::PRODUCT:
         case ttl::RATIO:
           {
             assert(node->tag != ttl::RATIO || node->b()->order() == 0);

             int l = map(node->a(), scalars, constants);
             int r = map(node->b(), scalars, constants);
             assert(l < r);
             assert(r == i - 1);

             stack.pop_back();
             stack.pop_back();
             record(node, top_of_stack, l);
           }
           break;

         case ttl::INDEX:
          assert(node->index.size() == 2);
          record(node, top_of_stack);
          break;

         case ttl::TENSOR:
          record(node, top_of_stack);
          map_tensor(node, scalars, constants);
          break;

         case ttl::RATIONAL:
          record(node, top_of_stack);;
          immediates[immediate++] = as<T>(node->q);
          break;

         case ttl::DOUBLE:
          record(node, top_of_stack);
          immediates[immediate++] = node->d;
          break;

         default:
          assert(false);
        }

        return i++;
      }
    };
  };
}
