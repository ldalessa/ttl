#pragma once

#include "ttl/Scalar.hpp"
#include "ttl/Tag.hpp"
#include "ttl/TensorTree.hpp"
#include "ttl/set.hpp"
#include <string_view>

namespace ttl
{
  namespace exec
  {
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

    constexpr bool is_binary(Tag tag)
    {
      return tag < IMMEDIATE;
    }

    struct Index
    {
      const char* i;
      const char* e;

      constexpr auto size() const -> int
      {
        return e - i;
      }

      constexpr char operator[](int index) const
      {
        return i[index];
      }

      constexpr friend bool operator==(Index const& a, Index const& b)
      {
        if (a.size() != b.size()) {
          return false;
        }

        for (int j = 0, e = a.size(); j < e; ++j)
        {
          if (a.i[j] != b.i[j]) {
            return false;
          }
        }

        return true;
      }
    };

    template <auto const& from, auto const& to>
    struct IndexMapper
    {
      constexpr auto operator()(std::array<int, from.size()> index) const
        -> std::array<int, to.size()>
      {
        if constexpr (from == to) {
          return index;
        }
        else {
          std::array<int, to.size()> out;
          for (int i = 0; i < to.size(); ++i) {
            out[i] = 0;
          }
          return out;
        }
      }
    };

    template <int N, std::size_t M>
    constexpr int row_major(std::array<int, M> const& index)
    {
      int sum = 0;
      int n = 1;
      for (int i : index) {
        if (i < 0 or N <= i) __builtin_unreachable();
        sum += n * i;
        n *= N;
      }
      return sum;
    }

    template <int N, std::size_t M, int m = 0>
    constexpr void eval(auto const& op, std::array<int, M> index = {}) {
      if constexpr (m == M) {
        op(index);
      }
      else {
        for (int i = 0; i < N; ++i) {
          index[m] = i;
          eval<N, M, m+1>(op, index);
        }
      }
    }
  }

  template <class T, int N, int Nodes, int Indices, int InnerIndices, int TensorIndices, int Scalars, int Immediates, int Stack>
  struct SerializedTensorTree
  {
    using Node = TensorTree::Node;
    using scalar_type = T;

    TreeShape const& shape;

    // Arrays of data.
    char        indices[Indices]{};
    char  inner_indices[InnerIndices]{};
    char tensor_indices[TensorIndices]{};
    int      scalar_ids[Scalars]{};
    T        immediates[Immediates]{};

    exec::Tag tags[Nodes]{};
    int        rvo[Nodes]{};
    int       left[Nodes]{};
    int      right[Nodes]{};

    int        index[Nodes + 1]{};
    int  inner_index[Nodes + 1]{};
    int tensor_index[Nodes + 1]{};
    int       scalar[Nodes + 1]{};
    int    immediate[Nodes + 1]{};

    constexpr SerializedTensorTree(TreeShape const& shape, TensorTree const& tree, set<Scalar> const& scalars, set<Scalar> const& constants)
        : shape { shape }
    {
      Builder_ builder(*this);
      builder.map(tree.root(), scalars, constants);
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

        // Variables used during the initialization process.
    struct Builder_
    {
      SerializedTensorTree& tree;

      int i = 0;
      int index = 0;
      int inner_index = 0;
      int tensor_index = 0;
      int scalar = 0;
      int immediate = 0;
      ce::dvector<int> stack;

      constexpr Builder_(SerializedTensorTree& tree)
          : tree(tree)
      {
        stack.reserve(Stack + 1);
        stack.push_back(0);
      }

      constexpr ~Builder_()
      {
        // The maps are all size+1, so that we can compute the size of each
        // extent for the root. This is standard compressed-sparse-row design.
        tree.index[i]        = std::size(tree.indices);
        tree.inner_index[i]  = std::size(tree.inner_indices);
        tree.tensor_index[i] = std::size(tree.tensor_indices);
        tree.scalar[i]       = std::size(tree.scalar_ids);
        tree.immediate[i]    = std::size(tree.immediates);

        assert(i == Nodes);
        assert(scalar == Scalars);
        assert(index == Indices);
        assert(inner_index == InnerIndices);
        assert(tensor_index == TensorIndices);
        assert(immediate == Immediates);
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
      constexpr void record(Node const* node, int tos)
      {
        tree.tags[i]         = to_tag(node);
        tree.rvo[i]          = tos;
        tree.index[i]        = index;
        tree.inner_index[i]  = inner_index;
        tree.tensor_index[i] = tensor_index;
        tree.scalar[i]       = scalar;
        tree.immediate[i]    = immediate;
        tree.left[i]         = -1;            // overwritten in map() for binary
        tree.right[i]        = -1;            // overwritten in map() for binary

        // Store my outer index to the right offset.
        for (char c : node->outer()) {
          tree.indices[index++] = c;
        }

        // Store the all index to the right offset.
        for (char c : node->all()) {
          tree.inner_indices[inner_index++] = c;
        }

        if (!std::is_constant_evaluated()) {
          const char* fmt = "{:<10}{:<10}{:<10}\n";
          if (i == 0) {
            fmt::print(fmt, "i", "index", "index_size");
          }
          int index_size = index - tree.index[i];
          fmt::print(fmt, i, tree.index[i], index_size);
        }
      }

      constexpr void map_tensor(Node const* node, set<Scalar> const& scalars, set<Scalar> const& constants)
      {
        // Store my scalar offsets to the scalar_ids array.
        node->scalars(N, [&](Scalar const& s) {
          if (s.constant) {
            auto i = constants.find(s);
            assert(i);
            tree.scalar_ids[scalar++] = *i;
          }
          else {
            auto i = scalars.find(s);
            assert(i);
            tree.scalar_ids[scalar++] = *i;
          }
        });

        // Store my tensor index
        for (char c : node->index) {
          tree.tensor_indices[tensor_index++] = c;
        }
      }

      constexpr int map(Node const* node, set<Scalar> const& scalars, set<Scalar> const& constants)
      {
        int tos = stack.back();
        stack.push_back(tos + node->tensor_size(N));

        switch (node->tag)
        {
         case ttl::SUM:
         case ttl::DIFFERENCE:
         case ttl::PRODUCT:
         case ttl::RATIO:
           {
             int l = map(node->a(), scalars, constants);
             int r = map(node->b(), scalars, constants);
             stack.pop_back();
             stack.pop_back();
             record(node, tos);
             tree.left[i]  = l;
             tree.right[i] = r;

             assert(l < r);
             assert(r == i - 1);
             assert(node->tag != ttl::RATIO || node->b()->order() == 0);
           }
           break;

         case ttl::INDEX:
          assert(node->index.size() == 2);
          record(node, tos);
          break;

         case ttl::TENSOR:
          record(node, tos);
          map_tensor(node, scalars, constants);
          break;

         case ttl::RATIONAL:
          record(node, tos);
          tree.immediates[immediate++] = as<T>(node->q);
          break;

         case ttl::DOUBLE:
          record(node, tos);
          tree.immediates[immediate++] = node->d;
          break;

         default:
          assert(false);
        }

        return i++;
      }
    };
  };

  template <class Tree,
            class Tags,
            class Indices,
            class IndexOffsets,
            class InnerIndices,
            class InnerIndexOffsets,
            class TensorIndices,
            class TensorIndexOffsets,
            class ScalarIds,
            class ScalarIdsOffsets,
            class Left>
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

    constexpr static std::array index_offsets = []<int... is>(std::integer_sequence<int, is...>) {
      return std::array{is...};
    }(IndexOffsets{});

    constexpr static std::array inner_indices = []<char... is>(std::integer_sequence<char, is...>) {
      return std::array{is...};
    }(InnerIndices{});

    constexpr static std::array inner_index_offsets = []<int... is>(std::integer_sequence<int, is...>) {
      return std::array{is...};
    }(InnerIndexOffsets{});

    constexpr static std::array tensor_indices = []<char... is>(std::integer_sequence<char, is...>) {
      return std::array{is...};
    }(TensorIndices{});

    constexpr static std::array tensor_index_offsets = []<int... is>(std::integer_sequence<int, is...>) {
      return std::array{is...};
    }(TensorIndexOffsets{});

    constexpr static std::array scalar_ids = []<int... ids>(std::integer_sequence<int, ids...>) {
      return std::array{ids...};
    }(ScalarIds{});

    constexpr static std::array scalar_ids_offsets = []<int... ids>(std::integer_sequence<int, ids...>) {
      return std::array{ids...};
    }(ScalarIdsOffsets{});

    constexpr static std::array left = []<int... ids>(std::integer_sequence<int, ids...>) {
      return std::array{ids...};
    }(Left{});

    constexpr static std::array right = []<int... ids>(std::integer_sequence<int, ids...>) {
      return std::array{(is_binary(tags[ids]) ? ids - 1 : -1)...};
    }(std::make_integer_sequence<int, std::size(tags)>());

    Tree tree;

    constexpr ExecutableTensorTree(Tree const& serialized_tree,
                                   Tags,
                                   Indices,
                                   IndexOffsets,
                                   InnerIndices,
                                   InnerIndexOffsets,
                                   TensorIndices,
                                   TensorIndexOffsets,
                                   ScalarIds,
                                   ScalarIdsOffsets,
                                   Left)
        : tree(serialized_tree)
    {
    }

    template <int k, auto const& index = indices, auto const& offsets = index_offsets>
    constexpr static auto make_index() -> exec::Index
    {
      static_assert(offsets[k] <= offsets[k+1]);
      return exec::Index {
        .i = &index[offsets[k]],
        .e = &index[offsets[k+1]]
      };
    }

    template <int k>
    void eval_sum(Stack& stack) const
    {
      // c = a + b
      constexpr static int l = left[k];
      constexpr static int r = right[k];

      static_assert(l < r);
      static_assert(r == k - 1);

      constexpr static exec::Index ci = make_index<k>();
      constexpr static exec::Index ai = make_index<l>();
      constexpr static exec::Index bi = make_index<r>();

      static_assert(ci == ai);

      constexpr static exec::IndexMapper<ci, bi> bmap{};

      constexpr static std::size_t M = ci.size();

      T* __restrict c = &stack[tree.rvo[k]];
      T* __restrict a = &stack[tree.rvo[l]];
      T* __restrict b = &stack[tree.rvo[r]];

      exec::eval<N, M>([&](std::array<int, M> outer) {
        int i = exec::row_major<N>(outer);
        int j = exec::row_major<N>(bmap(outer));
        c[i] = a[i] + b[j];
      });
    }

    template <int k>
    void eval_difference(Stack& stack) const
    {
      // c = a - b
      constexpr static int l = left[k];
      constexpr static int r = right[k];

      constexpr static exec::Index ci = make_index<k>();
      constexpr static exec::Index ai = make_index<l>();
      constexpr static exec::Index bi = make_index<r>();
      static_assert(ci == ai);

      constexpr static exec::IndexMapper<ci, bi> bmap{};

      constexpr static std::size_t M = ci.size();

      T* __restrict c = &stack[tree.rvo[k]];
      T* __restrict a = &stack[tree.rvo[l]];
      T* __restrict b = &stack[tree.rvo[r]];

      exec::eval<N, M>([&](std::array<int, M> outer) {
        int i = exec::row_major<N>(outer);
        int j = exec::row_major<N>(bmap(outer));
        c[i] = a[i] - b[j];
      });
    }

    template <int k>
    void eval_product(Stack& stack) const
    {
      // c = a * b
      constexpr static int l = left[k];
      constexpr static int r = right[k];

      constexpr static exec::Index  ci = make_index<k>();
      constexpr static exec::Index all = make_index<k, inner_indices, inner_index_offsets>();
      constexpr static exec::Index  ai = make_index<l>();
      constexpr static exec::Index  bi = make_index<r>();

      constexpr static exec::IndexMapper<all, ci> cmap{};
      constexpr static exec::IndexMapper<all, ai> amap{};
      constexpr static exec::IndexMapper<all, bi> bmap{};

      constexpr static int M = all.size();

      T* __restrict c = &stack[tree.rvo[k]];
      T* __restrict a = &stack[tree.rvo[l]];
      T* __restrict b = &stack[tree.rvo[r]];

      // Don't know the state of the stack but we're going to need to accumulate
      // there so we need to zero it first.
      for (int ii = 0; ii < ttl::pow(N, M); ++ii) {
        c[ii] = T();
      }

      exec::eval<N, M>([&](std::array<int, M> outer)
      {
        int ii = exec::row_major<N>(cmap(outer));
        int jj = exec::row_major<N>(amap(outer));
        int kk = exec::row_major<N>(bmap(outer));
        c[ii] += a[jj] * b[kk];
      });
    }

    template <int k>
    void eval_ratio(Stack& stack) const
    {
      // c = a / b
      constexpr static int l = left[k];
      constexpr static int r = right[k];

      // right now we're expecting `b` to be a scalar.
      constexpr static exec::Index  ci = make_index<k>();
      constexpr static exec::Index all = make_index<k, inner_indices, inner_index_offsets>();
      constexpr static exec::Index  ai = make_index<l>();
      constexpr static exec::Index  bi = make_index<r>();

      static_assert(ci == ai);
      static_assert(ci == all);
      static_assert(bi.size() == 0);

      constexpr static int M = ttl::pow(N, ci.size());

      T* __restrict c = &stack[tree.rvo[k]];
      T* __restrict a = &stack[tree.rvo[l]];
      T* __restrict b = &stack[tree.rvo[r]];

      auto rb = T(1)/b[0];
      for (int i = 0; i < M; ++i) {
        c[i] = a[i] * rb;
      }
    }

    template <int k>
    void eval_immediate(Stack& stack) const
    {
      stack[tree.rvo[k]] = tree.immediates[tree.immediate[k]];
    }

    template <int k>
    void eval_scalar(int i, Stack& stack, auto const& scalars) const
    {
      constexpr static exec::Index outer = make_index<k>();
      constexpr static exec::Index   all = make_index<k, inner_indices, inner_index_offsets>();
      constexpr static exec::Index index = make_index<k, tensor_indices, tensor_index_offsets>();

      constexpr static exec::IndexMapper<all, outer> outer_map{};
      constexpr static exec::IndexMapper<all, index> index_map{};

      constexpr static int M = all.size();
      constexpr static int const *ids = &scalar_ids[scalar_ids_offsets[k]];

      T* __restrict c = &stack[tree.rvo[k]];

      // Don't know the state of the stack but we're going to need to accumulate
      // there so we need to zero it first.
      for (int ii = 0; ii < ttl::pow(N, M); ++ii) {
        c[ii] = T();
      }

      exec::eval<N, M>([&](std::array<int, M> outer)
      {
        int ii = exec::row_major<N>(outer_map(outer));
        int jj = exec::row_major<N>(index_map(outer));
        c[ii] += scalars(ids[jj], i);
      });
    }

    template <int k>
    void eval_constant(Stack& stack, auto const& constants) const
    {
      constexpr static int const *ids = &scalar_ids[scalar_ids_offsets[k]];
      constexpr static int M = &scalar_ids[scalar_ids_offsets[k + 1]] - ids;

      T* __restrict c = &stack[tree.rvo[k]];

      for (int i = 0; i < M; ++i) {
        c[i] = constants(ids[i]);
      }
    }

    template <int k>
    void eval_delta(Stack& stack) const
    {
      T* __restrict c = &stack[tree.rvo[k]];

      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          c[i * N + j] = T(i == j);
        }
      }
    }

    template <int k>
    void eval_kernel_step(int i, Stack& stack, auto const& scalars, auto const& constants) const
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
        eval_constant<k>(stack, constants);
      }
      if constexpr (tags[k] == exec::SCALAR) {
        eval_scalar<k>(i, stack, scalars);
      }
      if constexpr (tags[k] == exec::DELTA) {
        eval_delta<k>(stack);
      }
    }

    void evaluate(auto const& scalars, auto const& constants) const
    {
      Stack stack{};
      [&]<std::size_t... i>(std::index_sequence<i...>) {
        (eval_kernel_step<i>(0, stack, scalars, constants), ...);
      }(std::make_index_sequence<Tree::n_nodes()>());
    }
  };
}
