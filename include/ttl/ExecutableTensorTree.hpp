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

  template <auto shape>
  struct SerializedTensorTree
  {
    using Node = TensorTree::Node;

    // for debugging
    decltype(shape) debug_shape = shape;

    // Arrays of data.
    char        indices[shape.n_indices];
    char  inner_indices[shape.n_inner_indices];
    char tensor_indices[shape.n_tensor_indices];
    int      scalar_ids[shape.n_scalars];
    // T        immediates[shape.n_immediates];

    exec::Tag tags[shape.n_nodes];
    int        rvo[shape.n_nodes];
    int       left[shape.n_nodes];
    int      right[shape.n_nodes];

    int        index_offsets[shape.n_nodes + 1];
    int  inner_index_offsets[shape.n_nodes + 1];
    int tensor_index_offsets[shape.n_nodes + 1];
    int   scalar_ids_offsets[shape.n_nodes + 1];
    int    immediate_offsets[shape.n_nodes + 1];

    constexpr SerializedTensorTree(TensorTree const& tree, set<Scalar> const& scalars, set<Scalar> const& constants)
    {
      Builder_ builder(*this);
      builder.map(tree.root(), scalars, constants);
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
        stack.reserve(shape.stack_depth + 1);
        stack.push_back(0);
      }

      constexpr ~Builder_()
      {
        // The maps are all size+1, so that we can compute the size of each
        // extent for the root. This is standard compressed-sparse-row design.
        tree.index_offsets[i]        = std::size(tree.indices);
        tree.inner_index_offsets[i]  = std::size(tree.inner_indices);
        tree.tensor_index_offsets[i] = std::size(tree.tensor_indices);
        tree.scalar_ids_offsets[i]   = std::size(tree.scalar_ids);
        tree.immediate_offsets[i]    = shape.n_immediates; // std::size(tree.immediates);

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
      constexpr void record(Node const* node, int top_of_stack)
      {
        assert(top_of_stack + node->tensor_size(shape.dims) <= shape.stack_depth);
        tree.tags[i]                 = to_tag(node);
        tree.rvo[i]                  = top_of_stack;
        tree.index_offsets[i]        = index;
        tree.inner_index_offsets[i]  = inner_index;
        tree.tensor_index_offsets[i] = tensor_index;
        tree.scalar_ids_offsets[i]   = scalar;
        tree.immediate_offsets[i]    = immediate;
        tree.left[i]                 = -1;            // overwritten in map() for binary
        tree.right[i]                = -1;            // overwritten in map() for binary

        // Store my outer index to the right offset.
        for (char c : node->outer()) {
          tree.indices[index++] = c;
        }

        // Store the all index to the right offset.
        for (char c : node->all()) {
          tree.inner_indices[inner_index++] = c;
        }
      }

      constexpr void map_tensor(Node const* node, set<Scalar> const& scalars, set<Scalar> const& constants)
      {
        // Store my scalar offsets to the scalar_ids array.
        node->scalars(shape.dims, [&](Scalar const& s) {
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
             int l = map(node->a(), scalars, constants);
             int r = map(node->b(), scalars, constants);
             stack.pop_back();
             stack.pop_back();
             record(node, top_of_stack);
             tree.left[i]  = l;
             tree.right[i] = r;

             assert(l < r);
             assert(r == i - 1);
             assert(node->tag != ttl::RATIO || node->b()->order() == 0);
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
          record(node, top_of_stack);
          immediate++;
          // tree.immediates[immediate++] = as<T>(node->q);
          break;

         case ttl::DOUBLE:
          record(node, top_of_stack);
          immediate++;
          // tree.immediates[immediate++] = node->d;
          break;

         default:
          assert(false);
        }

        return i++;
      }
    };
  };

  template <class T, auto shape, auto tree>
  struct ExecutableTensorTree
  {
    using Stack = T[shape.stack_depth];
    constexpr static int N = shape.dims;

    T immediates[shape.n_immediates];

    constexpr ExecutableTensorTree(TensorTree const& t)
    {
      collect_immediates(0, t.root());
    }

    constexpr void collect_immediates(int i, TensorTree::Node const* node)
    {
      if (tag_is_binary(node->tag)) {
        collect_immediates(i, node->a());
        collect_immediates(i, node->b());
      }
      else if (node->tag == ttl::RATIONAL) {
        immediates[i++] = as<T>(node->q);
      }
      else if (node->tag == ttl::DOUBLE) {
        immediates[i++] = node->d;
      }
    }

    template <int k>
    constexpr static auto make_index() -> exec::Index
    {
      static_assert(tree.index_offsets[k] <= tree.index_offsets[k+1]);
      return exec::Index {
        .i = &tree.indices[tree.index_offsets[k]],
        .e = &tree.indices[tree.index_offsets[k+1]]
      };
    }

    template <int k>
    constexpr static auto make_inner_index() -> exec::Index
    {
      static_assert(tree.inner_index_offsets[k] <= tree.inner_index_offsets[k+1]);
      return exec::Index {
        .i = &tree.inner_indices[tree.inner_index_offsets[k]],
        .e = &tree.inner_indices[tree.inner_index_offsets[k+1]]
      };
    }

    template <int k>
    constexpr static auto make_tensor_index() -> exec::Index
    {
      static_assert(tree.tensor_index_offsets[k] <= tree.tensor_index_offsets[k+1]);
      return exec::Index {
        .i = &tree.tensor_indices[tree.tensor_index_offsets[k]],
        .e = &tree.tensor_indices[tree.tensor_index_offsets[k+1]]
      };
    }

    template <int k>
    void eval_sum(Stack& stack) const
    {
      // c = a + b
      constexpr static int l = tree.left[k];
      constexpr static int r = tree.right[k];

      static_assert(l < r);
      static_assert(r == k - 1);

      constexpr static exec::Index ci = make_index<k>();
      constexpr static exec::Index ai = make_index<l>();
      constexpr static exec::Index bi = make_index<r>();

      static_assert(ci == ai);

      constexpr static exec::IndexMapper<ci, bi> bmap{};

      constexpr static int M = ci.size();

      constexpr static int rk = tree.rvo[k];
      constexpr static int rl = tree.rvo[l];
      constexpr static int rr = tree.rvo[r];

      static_assert(0 <= rk);
      static_assert(rk < rl);
      static_assert(rl < rr);
      static_assert(rr + ttl::pow(N, M) <= shape.stack_depth);

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

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
      constexpr static int l = tree.left[k];
      constexpr static int r = tree.right[k];

      constexpr static exec::Index ci = make_index<k>();
      constexpr static exec::Index ai = make_index<l>();
      constexpr static exec::Index bi = make_index<r>();
      static_assert(ci == ai);

      constexpr static exec::IndexMapper<ci, bi> bmap{};

      constexpr static int M = ci.size();

      constexpr static int rk = tree.rvo[k];
      constexpr static int rl = tree.rvo[l];
      constexpr static int rr = tree.rvo[r];

      static_assert(0 <= rk);
      static_assert(rk < rl);
      static_assert(rl < rr);
      static_assert(rr + ttl::pow(N, M) <= shape.stack_depth);

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

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
      constexpr static int l = tree.left[k];
      constexpr static int r = tree.right[k];

      static_assert(l < r);
      static_assert(r == k - 1);

      constexpr static exec::Index  ci = make_index<k>();
      constexpr static exec::Index all = make_inner_index<k>();
      constexpr static exec::Index  ai = make_index<l>();
      constexpr static exec::Index  bi = make_index<r>();

      constexpr static exec::IndexMapper<all, ci> cmap{};
      constexpr static exec::IndexMapper<all, ai> amap{};
      constexpr static exec::IndexMapper<all, bi> bmap{};

      constexpr static int rk = tree.rvo[k];
      constexpr static int rl = tree.rvo[l];
      constexpr static int rr = tree.rvo[r];

      static_assert(0 <= rk);
      static_assert(rk < rl);
      static_assert(rl < rr);
      static_assert(rr + ttl::pow(N, ci.size()) <= shape.stack_depth);

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      // Don't know the state of the stack but we're going to need to accumulate
      // there so we need to zero it first (it's nearly certainly dirty, either
      // from previous frame or from previous evaluation)
      for (int ii = 0; ii < ttl::pow(N, ci.size()); ++ii) {
        c[ii] = T();
      }

      constexpr static int M = all.size();
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
      constexpr static int l = tree.left[k];
      constexpr static int r = tree.right[k];

      // right now we're expecting `b` to be a scalar.
      constexpr static exec::Index  ci = make_index<k>();
      constexpr static exec::Index all = make_inner_index<k>();
      constexpr static exec::Index  ai = make_index<l>();
      constexpr static exec::Index  bi = make_index<r>();

      static_assert(ci == ai);
      static_assert(ci == all);
      static_assert(bi.size() == 0);

      constexpr static int M = ttl::pow(N, ci.size());

      constexpr static int rk = tree.rvo[k];
      constexpr static int rl = tree.rvo[l];
      constexpr static int rr = tree.rvo[r];

      static_assert(0 <= rk);
      static_assert(rk < rl);
      static_assert(rl < rr);
      static_assert(rr + M <= shape.stack_depth);

      T* const __restrict c = stack + rk;
      T* const __restrict a = stack + rl;
      T* const __restrict b = stack + rr;

      auto rb = T(1)/b[0];
      for (int i = 0; i < M; ++i) {
        c[i] = a[i] * rb;
      }
    }

    template <int k>
    void eval_immediate(Stack& stack) const
    {
      stack[tree.rvo[k]] = immediates[tree.immediate_offsets[k]];
    }

    template <int k>
    void eval_scalar(int i, Stack& stack, auto const& scalars) const
    {
      constexpr static exec::Index outer = make_index<k>();
      constexpr static exec::Index   all = make_inner_index<k>();
      constexpr static exec::Index index = make_tensor_index<k>();

      constexpr static exec::IndexMapper<all, outer> outer_map{};
      constexpr static exec::IndexMapper<all, index> index_map{};

      constexpr static int rk = tree.rvo[k];

      static_assert(0 <= rk);
      static_assert(rk + ttl::pow(N, outer.size()) <= shape.stack_depth);

      constexpr static int const *ids = &tree.scalar_ids[tree.scalar_ids_offsets[k]];

      T* const __restrict c = stack + rk;

      // Don't know the state of the stack but we're going to need to accumulate
      // there so we need to zero it first (it's nearly certainly dirty, either
      // from previous frame or from previous evaluation).
      for (int ii = 0; ii < ttl::pow(N, outer.size()); ++ii) {
        c[ii] = T();
      }

      constexpr static int M = all.size();
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
      constexpr static int const *ids = &tree.scalar_ids[tree.scalar_ids_offsets[k]];
      constexpr static int M = &tree.scalar_ids[tree.scalar_ids_offsets[k + 1]] - ids;

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
      if constexpr (tree.tags[k] == exec::SUM) {
        eval_sum<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::DIFFERENCE) {
        eval_difference<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::PRODUCT) {
        eval_product<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::RATIO) {
        eval_ratio<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::IMMEDIATE) {
        eval_immediate<k>(stack);
      }
      if constexpr (tree.tags[k] == exec::CONSTANT) {
        eval_constant<k>(stack, constants);
      }
      if constexpr (tree.tags[k] == exec::SCALAR) {
        eval_scalar<k>(i, stack, scalars);
      }
      if constexpr (tree.tags[k] == exec::DELTA) {
        eval_delta<k>(stack);
      }
    }

    void evaluate(auto const& scalars, auto const& constants) const
    {
      Stack stack{};
      [&]<std::size_t... i>(std::index_sequence<i...>) {
        (eval_kernel_step<i>(0, stack, scalars, constants), ...);
      }(std::make_index_sequence<shape.n_nodes>());
    }
  };
}
