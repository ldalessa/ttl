#pragma once

#include "ttl/Scalar.hpp"
#include "ttl/Tag.hpp"
#include "ttl/TensorTree.hpp"
#include "ttl/TreeShape.hpp"
#include "ttl/exec.hpp"
#include "ttl/pack_fp.hpp"
#include "ttl/set.hpp"
#include <array>
#include <vector>

namespace ttl
{
  template <class T, TreeShape shape>
  struct SerializedTree
  {
    using serialized_tree_tag = void;
    using Node = TensorTree::Node;

    // For debugging (this is only needed for gdb, so that I can actually
    // inspect the shape that we're dealing with, the CNTTP one is obliterated.
    decltype(shape) debug_shape = shape;

    // Arrays of compressed data.
    std::array<char,shape.n_indices>        indices_;        //!< `ij` outer index
    std::array<char,shape.n_inner_indices>  inner_indices_;  //!< `ij` contracted
    std::array<char,shape.n_tensor_indices> tensor_indices_; //!< `iij` tensor
    std::array<int,shape.n_scalars>         scalar_ids_;     //!< scalar ids for tensors
    std::array<uint64_t,shape.n_immediates> immediates_;     //!< just double in gcc-11
    std::array<char,shape.n_tensor_ids>     tensor_ids_;

    // Per-node state.
    std::array<exec::Tag,shape.n_nodes> tags ;             //!< type of each node
    std::array<int,shape.n_nodes>        rvo_;             //!< return stack slot
    std::array<int,shape.n_nodes>       left_;             //!< index of left child (if any)
    std::array<int,shape.n_nodes>      order_;             //!< tensor order (if any)

    // Per-node offsets into the compressed data.
    std::array<int,shape.n_nodes + 1>        index_offsets_;
    std::array<int,shape.n_nodes + 1>  inner_index_offsets_;
    std::array<int,shape.n_nodes + 1> tensor_index_offsets_;
    std::array<int,shape.n_nodes + 1>   scalar_ids_offsets_;
    std::array<int,shape.n_nodes + 1>    immediate_offsets_;
    std::array<int,shape.n_nodes + 1>   tensor_ids_offsets_;

    /// Create a serialized tree from a tensor tree
    constexpr SerializedTree(TensorTree const& tree,
                             set<Scalar> const& scalars,
                             set<Scalar> const& constants)
    {
      {
        Builder_ builder(*this);
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
        assert(immediate_offsets_[i]    <= immediate_offsets_[i+1]);
        assert(tensor_ids_offsets_[i]   <= tensor_ids_offsets_[i+1]);

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

    constexpr double immediate(int k) const
    {
      return unpack_fp(immediates_[immediate_offsets_[k]]);
    }

    constexpr Tensor tensor(int k) const
    {
      assert(tags[k] == exec::CONSTANT || tags[k] == exec::SCALAR);
      std::string_view id {
        &tensor_ids_[tensor_ids_offsets_[k]],
        &tensor_ids_[tensor_ids_offsets_[k+1]],
      };
      return Tensor(id, order_[k]);
    }

    constexpr int n_constant_coefficients() const
    {
      int n = 0;
      for (exec::Tag tag : tags) {
        n += (tag == exec::CONSTANT);
      }
      return n;
    }

    constexpr void get_scalars(bool constant, set<Scalar>& scalars) const
    {
      for (int i = 0; i < shape.n_nodes; ++i)
      {
        exec::Tag tag = tags[i];
        if (( constant && tag == exec::CONSTANT) || (!constant && tag == exec::SCALAR))
        {
          Tensor          t = tensor(i);
          exec::Index  from = inner_index(i);
          exec::Index    to = tensor_index(i);
          ScalarIndex index(from.size());
          do {
            scalars.emplace(t, index.select(from, to), constant, shape.dims);
          } while (index.carry_sum_inc(shape.dims));
        }
      }
    }

    // Variables used during the initialization process.
    struct Builder_
    {
      SerializedTree& tree;

      int i = 0;
      int index = 0;
      int inner_index = 0;
      int tensor_index = 0;
      int tensor = 0;
      int scalar = 0;
      int immediate = 0;
      std::vector<int> stack;

      constexpr Builder_(SerializedTree& tree)
          : tree(tree)
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
        tree.immediate_offsets_[i]    = std::size(tree.immediates_);
        tree.tensor_ids_offsets_[i]   = std::size(tree.tensor_ids_);

        assert(i == shape.n_nodes);
        assert(scalar == shape.n_scalars);
        assert(tensor == shape.n_tensor_ids);
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
        tree.tags[i]                  = to_tag(node);
        tree.rvo_[i]                  = top_of_stack;
        tree.left_[i]                 = left;
        tree.order_[i]                = node->order();

        tree.index_offsets_[i]        = index;
        tree.inner_index_offsets_[i]  = inner_index;
        tree.tensor_index_offsets_[i] = tensor_index;
        tree.scalar_ids_offsets_[i]   = scalar;
        tree.immediate_offsets_[i]    = immediate;
        tree.tensor_ids_offsets_[i]   = tensor;

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
        tree.order_[i] = node->tensor.order();

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

        // Store my tensor.
        for (char c : node->tensor.id()) {
          tree.tensor_ids_[tensor++] = c;
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
          tree.immediates_[immediate++] = pack_fp(as<T>(node->q));
          break;

         case ttl::DOUBLE:
          record(node, top_of_stack);
          tree.immediates_[immediate++] = pack_fp(node->d);
          break;

         default:
          assert(false);
        }

        return i++;
      }
    };
  };

  template <class T>
  concept serialized_tree = requires {
    typename std::remove_cvref_t<T>::serialized_tree_tag;
  };
}
