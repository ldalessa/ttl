#pragma once

#include "ttl/Scalar.hpp"
#include "ttl/Tag.hpp"
#include "ttl/TensorTree.hpp"
#include "ttl/set.hpp"

namespace ttl
{
  namespace exec {
    enum Tag : int {
      SUM,
      DIFFERENCE,
      PRODUCT,
      RATIO,
      IMMEDIATE,
      SCALAR,
      CONSTANT,
      DELTA
    };
  }

  template <class T, int N, int Nodes, int Indices, int Scalars, int Immediates, int Stack>
  struct SerializedTensorTree
  {
    using Node = TensorTree::Node;
    using scalar_type = T;

    TreeShape const& shape;

    // Arrays of data.
    char    indices[Indices]{};
    int  scalar_ids[Scalars]{};
    T    immediates[Immediates]{};

    exec::Tag tags[Nodes]{};
    int        rvo[Nodes]{};
    int      index[Nodes]{};
    int     scalar[Nodes]{};
    int  immediate[Nodes]{};
    int       left[Nodes]{};
    int      right[Nodes]{};

    // Variables used during the initialization process.
    int i = 0;
    int i_index = 0;
    int i_scalar = 0;
    int i_immediate = 0;
    ce::dvector<int> stack;

    constexpr SerializedTensorTree(TreeShape const& shape, TensorTree const& tree, set<Scalar> const& scalars, set<Scalar> const& constants)
        : shape { shape }
        , stack { std::in_place, 0 }
    {
      stack.reserve(shape.stack_depth + 1);
      map_(tree.root(), scalars, constants);
      assert(i == Nodes);
      assert(i_scalar == Scalars);
      assert(i_index == Indices);
      assert(i_immediate == Immediates);
      assert(stack.size() == 2);
      assert(stack.back() == tree.root()->tensor_size(N));
      stack.resize(0);
      stack.shrink_to_fit();
    }

    static constexpr int dims()
    {
      return N;
    }

    static constexpr int n_nodes()
    {
      return Nodes;
    }

    static constexpr int stack_size()
    {
      return Stack;
    }

    constexpr exec::Tag to_tag(Node const* tree)
    {
      switch (tree->tag)
      {
       case ttl::SUM:        return exec::SUM;
       case ttl::DIFFERENCE: return exec::DIFFERENCE;
       case ttl::PRODUCT:    return exec::PRODUCT;
       case ttl::RATIO:      return exec::RATIO;
       case ttl::INDEX:      return exec::DELTA;
       case ttl::TENSOR:     return (tree->constant) ? exec::CONSTANT : exec::SCALAR;
       case ttl::RATIONAL:   return exec::IMMEDIATE;
       case ttl::DOUBLE:     return exec::IMMEDIATE;
       default:
        assert(false);
      }
      __builtin_unreachable();
    }

    /// Record the information associated with a leaf node
    constexpr void record_(Node const* tree, int tos)
    {
      tags[i]      = to_tag(tree);
      rvo[i]       = tos;
      index[i]     = i_index;
      scalar[i]    = i_scalar;
      immediate[i] = i_immediate;

      // Store my index to the right offset.
      for (char id : tree->index) {
        indices[i_index++] = id;
      }
    }

    constexpr int map_(Node const* tree, set<Scalar> const& scalars, set<Scalar> const& constants)
    {
      int tos = stack.back();
      stack.push_back(tos + tree->tensor_size(N));

      switch (tree->tag)
      {
       case ttl::SUM:
       case ttl::DIFFERENCE:
       case ttl::PRODUCT:
       case ttl::RATIO:
         {
           int l = map_(tree->a(), scalars, constants);
           int r = map_(tree->b(), scalars, constants);
           stack.pop_back();
           stack.pop_back();
           record_(tree, tos);
           left[i]  = l;
           right[i] = r;
         }
        break;

       case ttl::INDEX:
        assert(tree->index.size() == 2);
        record_(tree, tos);
        break;

       case ttl::TENSOR:
        record_(tree, tos);
        map_tensor_(tree, scalars, constants);
        break;

       case ttl::RATIONAL:
        record_(tree, tos);
        immediates[i_immediate++] = as<T>(tree->q);
        break;

       case ttl::DOUBLE:
        record_(tree, tos);
        immediates[i_immediate++] = tree->d;
        break;

       default:
        assert(false);
      }

      return i++;
    }

    constexpr void map_tensor_(Node const* tree, set<Scalar> const& scalars, set<Scalar> const& constants)
    {
      // Store my scalar offsets to the scalar_ids array.
      tree->scalars(N, [&](Scalar const& s) {
        if (s.constant) {
          auto i = constants.find(s);
          assert(i);
          scalar_ids[i_scalar++] = *i;
        }
        else {
          auto i = scalars.find(s);
          assert(i);
          scalar_ids[i_scalar++] = *i;
        }
      });
    }
  };

  template <class Tree, class Tags, class Indices, class ScalarIds>
  struct ExecutableTensorTree
  {
    using T = typename Tree::scalar_type;
    using Stack = T[Tree::stack_size()];

    constexpr static int N = Tree::dims();

    constexpr static std::array tags = []<auto... tags>(std::integer_sequence<exec::Tag, tags...>) {
      return std::array{tags...};
    }(Tags{});

    constexpr static std::array indices = []<char... is>(std::integer_sequence<char, is...>) {
      return std::array{is...};
    }(Indices{});

    constexpr static std::array scalar_ids = []<int... ids>(std::integer_sequence<int, ids...>) {
      return std::array{ids...};
    }(ScalarIds{});

    Tree tree;

    constexpr ExecutableTensorTree(Tree const& serialized_tree, Tags, Indices, ScalarIds)
        : tree(serialized_tree)
    {
    }

    template <int k>
    void eval_sum(Stack& stack) const
    {
      // c = a + b
      T* __restrict c = &stack[tree.rvo[k]];
      T* __restrict a = &stack[tree.rvo[tree.left[k]]];
      T* __restrict b = &stack[tree.rvo[tree.right[k]]];

    }

    template <int k>
    void eval_difference(Stack& stack) const
    {
      // c = a + b
      T* __restrict c = &stack[tree.rvo[k]];
      T* __restrict a = &stack[tree.rvo[tree.left[k]]];
      T* __restrict b = &stack[tree.rvo[tree.right[k]]];
    }

    template <int k>
    void eval_product(Stack& stack) const
    {
      // c = a * b
      T* __restrict c = &stack[tree.rvo[k]];
      T* __restrict a = &stack[tree.rvo[tree.left[k]]];
      T* __restrict b = &stack[tree.rvo[tree.right[k]]];

    }

    template <int k>
    void eval_ratio(Stack& stack) const
    {
      // c = a / b
      T* __restrict c = &stack[tree.rvo[k]];
      T* __restrict a = &stack[tree.rvo[tree.left[k]]];
      T* __restrict b = &stack[tree.rvo[tree.right[k]]];

    }

    template <int k>
    void eval_immediate(Stack& stack) const
    {
      stack[tree.rvo[k]] = tree.immediates[tree.immediate[k]];
    }

    template <int k>
    void eval_scalar(Stack& stack) const
    {
    }

    template <int k>
    void eval_constant(Stack& stack) const
    {
    }

    template <int k>
    void eval_delta(Stack& stack) const
    {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          stack[tree.rvo[k] + i * N + j] = T(i == j);
        }
      }
    }

    template <int k>
    void eval_kernel_step(Stack& stack) const
    {
      if constexpr (tags[k] == exec::SUM) {
        eval_sum<k>(stack);
      }
      if constexpr (tags[k] == exec::DIFFERENCE) {
        eval_difference<k>(stack);
      }
      if constexpr (tags[k] == exec::PRODUCT) {
        eval_product<k>(stack);
      }
      if constexpr (tags[k] == exec::RATIO) {
        eval_ratio<k>(stack);
      }
      if constexpr (tags[k] == exec::IMMEDIATE) {
        eval_immediate<k>(stack);
      }
      if constexpr (tags[k] == exec::CONSTANT) {
        eval_constant<k>(stack);
      }
      if constexpr (tags[k] == exec::SCALAR) {
        eval_scalar<k>(stack);
      }
      if constexpr (tags[k] == exec::DELTA) {
        eval_delta<k>(stack);
      }
    }

    void evaluate() const
    {
      [this]<std::size_t... i>(std::index_sequence<i...>) {
        Stack stack;
        (eval_kernel_step<i>(stack), ...);
      }(std::make_index_sequence<Tree::n_nodes()>());
    }
  };
}
