#pragma once

#include "Partial.hpp"
#include "TensorTree.hpp"
#include "utils.hpp"
#include <array>

namespace ttl {
struct ScalarTree
{
  enum ScalarTag {
    BINARY,
    SCALAR,
    CONSTANT,
    IMMEDIATE
  };

  struct Node {
    ScalarTag tag;
    const Node* a = nullptr;
    const Node* b = nullptr;
    union {
      struct {} _monostate = {};
      Tag binary;
      int offset;
      double   d;
    };

    constexpr Node(Tag t, const Node* a, const Node* b)
        : tag(BINARY)
        , a(a)
        , b(b)
        , binary(t)
    {
      assert(ttl::binary(t));
    }

    constexpr Node(ScalarTag t, int offset)
        : tag(t)
        , offset(offset)
    {
      assert(tag == SCALAR || tag == CONSTANT);
    }

    constexpr Node(double value)
        : tag(IMMEDIATE)
        , d(value)
    {
    }

    constexpr bool is_zero() const {
      return (tag == IMMEDIATE && d == 0);
    }

    constexpr bool is_one() const {
      return (tag == IMMEDIATE && d == 1);
    }
  };

  template <typename Tree, typename Scalars, typename Constants>
  struct Builder {
    const Tree& tree;
    const Scalars& scalars;
    const Constants& constants;
    int N;

    using TreeIndex = ce::dvector<int>;

    constexpr Builder(const Tree& tree, const Scalars& scalars,
                      const Constants& constants)
        : tree(tree)
        , scalars(scalars)
        , constants(constants)
        , N(scalars.dim())
    {}

    constexpr ce::dvector<const Node*> operator()() const
    {
      int order = tree.order();
      ce::dvector<const Node*> out;
      TreeIndex index(order);
      do {
        out.push_back(handle(tree.root(), index));
      } while (utils::carry_sum_inc(N, 0, order, index));
      return out;
    }

    constexpr const Node*
    handle(const TensorTreeNode& node, const TreeIndex& index) const
    {
      switch (node.tag) {
       case SUM:
       case DIFFERENCE: return sum(node, index);
       case PRODUCT:
       case RATIO:      return product(node, index);
       case BIND:
       case PARTIAL:    return bind(node, index);
        // case INDEX:
       case DELTA:      return delta(index);
       case TENSOR:     return tensor(node.tensor(), index);
       case DOUBLE:     return new Node(node.d());
       case RATIONAL:   return new Node(to_double(node.q()));
       default:         assert(false);
      }
    }

    constexpr static const Node*
    combine(Tag tag, const Node* a, const Node* b) {
      if (tag == SUM && a->is_zero()) {
        delete a;
        return b;
      }

      if (tag == SUM && b->is_zero()) {
        delete b;
        return a;
      }

      if (tag == DIFFERENCE && a->is_zero()) {
        delete a;
        return combine(PRODUCT, new Node(-1), b);
      }

      if (tag == DIFFERENCE && b->is_zero()) {
        delete b;
        return a;
      }

      if (tag == PRODUCT && a->is_zero()) {
        delete b;
        return a;
      }

      if (tag == PRODUCT && b->is_zero()) {
        delete a;
        return b;
      }

      if (tag == PRODUCT && a->is_one()) {
        delete a;
        return b;
      }

      if (tag == PRODUCT && b->is_one()) {
        delete b;
        return a;
      }

      if (tag == RATIO) {
        assert(!b->is_zero());
      }

      if (tag == RATIO && a->is_zero()) {
        delete b;
        return a;
      }

      if (tag == RATIO && b->is_one()) {
        delete b;
        return a;
      }

      if (a->tag == IMMEDIATE && b->tag == IMMEDIATE) {
        Node* n = new Node(apply(tag, a->d, b->d));
        delete a;
        delete b;
        return n;
      }

      // optimize the tree by performing the inverse at transformation time
      if (tag == RATIO && b->tag != BINARY) {
        const Node* inverse = combine(RATIO, new Node(1), b);
        return new Node(PRODUCT, a, inverse);
      }

      return new Node(tag, a, a);
    }

    constexpr const Node*
    sum(const TensorTreeNode& node, const TreeIndex& index) const
    {
      const TensorTreeNode& a = tree.a(node);
      const TensorTreeNode& b = tree.b(node);
      Index outer = node.outer();
      const Node* l = handle(a, select(index, outer, a.outer()));
      const Node* r = handle(b, select(index, outer, b.outer()));
      return combine(node.tag, l, r);
    }

    constexpr const Node*
    product(const TensorTreeNode& node, const TreeIndex& index) const
    {
      const TensorTreeNode& a = tree.a(node);
      const TensorTreeNode& b = tree.b(node);

      if (node.tag == RATIO) {
        assert(b.order() == 0); // can't handle arbitrary inverse right now
      }

      Index outer = node.outer();
      Index inner = a.outer() & b.outer();
      Index   all = outer + inner;

      return contract(all, index, [&] (const TreeIndex& inner) -> const Node* {
        const Node* l = handle(a, select(inner, all, a.outer()));
        const Node* r = handle(b, select(inner, all, b.outer()));
        return combine(node.tag, l, r);
      });
    }

    constexpr const Node*
    bind(const TensorTreeNode& node, const TreeIndex& index) const
    {
      const TensorTreeNode& a = tree.a(node);
      const TensorTreeNode& b = tree.b(node);
      assert(b.tag == INDEX);

      Index outer = node.outer(); assert(outer == (a.outer() ^ b.outer()));
      Index inner = a.outer() & b.outer();
      Index   all = outer + inner;

      return contract(all, index, [&] (const TreeIndex& inner) -> const Node* {
        return handle(a, select(inner, all, a.outer()));
      });
    }

    constexpr const Node*
    delta(const TreeIndex& index) const
    {
      if (int e = index.size()) {
        int n = index[0];
        for (int i = 1; i < e; ++i) {
          if (index[i] != n) {
            return new Node(0);
          }
        }
      }
      return new Node(1);
    }

    constexpr const Node*
    tensor(const Tensor* t, const TreeIndex& index) const
    {
      if (auto&& i = utils::index_of(constants, t)) {
        return new Node(CONSTANT, *i);
      }
      return new Node(SCALAR, scalars.find(t, index));
    }

    constexpr static TreeIndex
    select(const TreeIndex& index, Index from, Index to)
    {
      assert(index.size() == from.size());
      // assert(from.size() == to.size());
      TreeIndex out;
      out.reserve(to.size());
      for (char c : to) {
        out.push_back(index[*from.index_of(c)]);
      }
      return out;
    }

    template <typename Op>
    constexpr const Node*
    contract(const Index& all, const TreeIndex& index, Op&& op) const
    {
      int n = index.size();
      int order = all.size();

      TreeIndex inner(order);
      for (int i = 0; i < n; ++i) {
        inner[i] = index[i];
      }

      // return a sum of ops
      const Node* out = new Node(0);
      do {
        out = combine(SUM, out, op(inner));
      } while (utils::carry_sum_inc(N, n, order, inner));
      return out;
    }
  };

  ce::dvector<const Node*> roots;

  template <typename Tree, typename Scalars, typename Constants>
  constexpr ScalarTree(const Tree& tree, const Scalars& scalars,
                       const Constants& constants)
  {
    roots = Builder(tree, scalars, constants)();
  }
};
}
