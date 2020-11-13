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
    Node* a = nullptr;
    Node* b = nullptr;
    union {
      struct {} _monostate = {};
      Tag binary;
      int offset;
      double   d;
    };

    constexpr Node(Tag t, Node* a, Node* b)
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

    constexpr friend bool operator==(const Node& a, const Node& b) {
      if (a.tag != b.tag) return false;
      if (a.tag == IMMEDIATE) return a.d == b.d;
      if (a.tag != BINARY)    return a.offset == b.offset;
      return *a.a == *b.b;
    }

    constexpr bool is_zero() const {
      return (tag == IMMEDIATE && d == 0);
    }

    constexpr bool is_one() const {
      return (tag == IMMEDIATE && d == 1);
    }

    constexpr bool is_binary(Tag t) const {
      return (tag == BINARY && binary == t);
    }

    constexpr Node* unlink() {
      assert(tag == BINARY);
      a = nullptr;
      b = nullptr;
      return this;
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

    // outer dimension tree generation
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

    // generic node dispatch
    constexpr Node*
    handle(const TensorTreeNode& node, const TreeIndex& index) const
    {
      switch (node.tag) {
       case SUM:
       case DIFFERENCE: return sum(node, index);
       case PRODUCT:
       case RATIO:      return product(node, index);
       case BIND:
       case PARTIAL:    return partial(node, index);
       case DELTA:      return delta(index);
       case TENSOR:     return tensor(node, index);
       case DOUBLE:     return new Node(node.d());
       case RATIONAL:   return new Node(to_double(node.q()));
        // case INDEX: right child of bind and partial isn't traversed
       default:         assert(false);
      }
    }

    // simplification algorithms.
    constexpr static Node*
    combine(Tag tag, Node* a, Node* b)
    {
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

      // immediately evaluate immediates
      if (a->tag == IMMEDIATE && b->tag == IMMEDIATE) {
        Node* n = new Node(apply(tag, a->d, b->d));
        delete a;
        delete b;
        return n;
      }

      // optimize the tree by performing the inverse at transformation time
      if (tag == RATIO && b->tag == IMMEDIATE) {
        Node* inverse = new Node(1.0 / b->d);
        return combine(PRODUCT, a, inverse);
      }

      // reduce the tree when a == b
      if (*a == *b) {
        if (tag == SUM) {
          delete b;
          return combine(PRODUCT, new Node(2), a);
        }
        if (tag == DIFFERENCE) {
          delete a;
          delete b;
          return new Node(0);
        }
        if (tag == RATIO) {
          delete a;
          delete b;
          return new Node(1);
        }
      }

      if (tag == PRODUCT && a->tag == IMMEDIATE && b->is_binary(PRODUCT)) {
        Node* ba = b->a;
        if (ba->tag == IMMEDIATE) {
          Node* bb = b->b;
          double d = a->d * ba->d;
          b->unlink();
          delete a;
          delete b;
          return combine(PRODUCT, new Node(d), bb);
        }
        return new Node(PRODUCT, a, b);
      }

      if (tag == PRODUCT && a->is_binary(PRODUCT) && b->tag == IMMEDIATE) {
        Node* aa = a->a;
        if (aa->tag == IMMEDIATE) {
          Node* ab = a->b;
          double d = aa->d * b->d;
          a->unlink();
          delete a;
          delete b;
          return combine(PRODUCT, new Node(d), ab);
        }
        return new Node(PRODUCT, b, a);
      }

      if (tag == PRODUCT && a->is_binary(PRODUCT) && b->is_binary(PRODUCT)) {
        Node* aa = a->a;
        Node* ab = a->b;
        Node* ba = b->a;
        Node* bb = b->b;
        if (aa->tag == IMMEDIATE && ba->tag == IMMEDIATE) {
          delete a->unlink();
          delete b->unlink();
          double d = aa->d * ba->d;
          a = new Node(d);
          b = new Node(PRODUCT, ab, bb);
          return combine(PRODUCT, a, b);
        }
        if (aa->tag == IMMEDIATE) {
          delete a->unlink();
          b = new Node(PRODUCT, ab, b);
          return new Node(PRODUCT, aa, b);
        }
        if (ba->tag == IMMEDIATE) {
          delete b->unlink();
          a = new Node(PRODUCT, bb, a);
          return new Node(PRODUCT, ba, a);
        }
      }

      return new Node(tag, a, b);
    }

    constexpr Node*
    sum(const TensorTreeNode& node, const TreeIndex& index) const
    {
      const TensorTreeNode& a = tree.a(node);
      const TensorTreeNode& b = tree.b(node);
      Index outer = node.outer();
      Node* l = handle(a, select(index, outer, a.outer()));
      Node* r = handle(b, select(index, outer, b.outer()));
      return combine(node.tag, l, r);
    }

    constexpr Node*
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

      return contract(all, index, [&] (const TreeIndex& inner) -> Node* {
        Node* l = handle(a, select(inner, all, a.outer()));
        Node* r = handle(b, select(inner, all, b.outer()));
        return combine(node.tag, l, r);
      });
    }

    constexpr Node*
    partial(const TensorTreeNode& node, const TreeIndex& index) const
    {
      const TensorTreeNode& a = tree.a(node);
      const TensorTreeNode& b = tree.b(node);
      assert(a.tag == TENSOR);
      assert(b.tag == INDEX);

      Index outer = node.outer();
      Index inner = b.outer();
      Index   all = outer + repeated(inner);

      return contract(all, index, [&] (const TreeIndex& inner) -> Node* {
        return tensor(a, select(inner, all, b.outer()));
      });
    }

    constexpr Node*
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

    constexpr Node*
    tensor(const TensorTreeNode& node, const TreeIndex& index) const
    {
      const Tensor* t = node.tensor();
      assert(t);
      if (auto&& i = constants.find(t, index)) {
        return new Node(CONSTANT, *i);
      }
      return new Node(SCALAR, *scalars.find(t, index));
    }

    constexpr static TreeIndex
    select(const TreeIndex& index, Index from, Index to)
    {
      assert(index.size() == from.size());
      TreeIndex out;
      out.reserve(to.size());
      for (char c : to) {
        out.push_back(index[*from.index_of(c)]);
      }
      return out;
    }

    template <typename Op>
    constexpr Node*
    contract(const Index& all, const TreeIndex& index, Op&& op) const
    {
      int n = index.size();
      int order = all.size();

      TreeIndex inner(order);
      for (int i = 0; i < n; ++i) {
        inner[i] = index[i];
      }

      // return a sum of ops
      Node* out = new Node(0);
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

template <>
struct fmt::formatter<ttl::ScalarTree::Node> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::ScalarTree::Node& node, FormatContext& ctx)
  {
    switch (node.tag) {
     case ttl::ScalarTree::BINARY:    return format_to(ctx.out(), "{}", node.binary);
     case ttl::ScalarTree::SCALAR:    return format_to(ctx.out(), "scalars[{}]", node.offset);
     case ttl::ScalarTree::CONSTANT:  return format_to(ctx.out(), "constants[{}]", node.offset);
     case ttl::ScalarTree::IMMEDIATE: return format_to(ctx.out(), "{}", node.d);
    };
    __builtin_unreachable();
  }
};

template <>
struct fmt::formatter<ttl::ScalarTree> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::ScalarTree& tree, FormatContext& ctx)
  {
    int i = 0;
    auto op = [&](const ttl::ScalarTree::Node* node, auto&& self) -> int {
      if (node->tag == ttl::ScalarTree::BINARY) {
        int a = self(node->a, self);
        int b = self(node->b, self);
        format_to(ctx.out(), "\tnode{}[label=\"{}\"]\n", i, *node);
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, b);
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, a);
      }
      else {
        format_to(ctx.out(), "\tnode{}[label=\"{}\"]\n", i, *node);
      }
      return i++;
    };

    for (const ttl::ScalarTree::Node* tree : tree.roots) {
      op(tree, op);
    }
    return ctx.out();
  }
};
