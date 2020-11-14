#pragma once

#include "Partial.hpp"
#include "TensorTree.hpp"
#include "utils.hpp"
#include <ce/dvector.hpp>
#include <concepts>

namespace ttl {
struct ScalarTreeNode
{
  enum Op {
    LEAF = 0,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE
  };

  constexpr static const char* op_to_string[] = {
    "",
    "+",
    "-",
    "*",
    "/"
  };

  constexpr static Op tag_to_op(Tag tag) {
    switch (tag) {
     case SUM:        return ADD;
     case DIFFERENCE: return SUBTRACT;
     case PRODUCT:    return MULTIPLY;
     case RATIO:      return DIVIDE;
     case BIND:
     case PARTIAL:
     case INDEX:
     case DELTA:      assert(false);
     case TENSOR:
     case RATIONAL:
     case DOUBLE:     return LEAF;
    }
    __builtin_unreachable();
  }


  Op             op = LEAF;
  double          d = 1.0;
  Rational        q = {1};
  bool     constant = false;
  int        offset = -1;
  ScalarTreeNode* a = nullptr;
  ScalarTreeNode* b = nullptr;

  constexpr ScalarTreeNode() = default;

  constexpr ScalarTreeNode(bool constant, int offset)
      : constant(constant)
      , offset(offset)
  {
  }

  constexpr ScalarTreeNode(std::floating_point auto d) : d(d) {
  }

  constexpr ScalarTreeNode(std::signed_integral auto i) : q(i) {
  }

  constexpr ScalarTreeNode(Rational q) : q(q) {
  }

  constexpr ScalarTreeNode(Op op, ScalarTreeNode* a, ScalarTreeNode* b)
      : op(op)
      , a(a)
      , b(b)
  {
    assert(op != LEAF && a && b);
  }

  constexpr static bool equivalent(ScalarTreeNode* a, ScalarTreeNode* b) {
    assert(a && b);
    if (a->op != b->op) return false;
    if (a->d != b->d) return false;
    if (a->q != b->q) return false;
    if (a->constant != b->constant) return false;
    if (a->offset != b->offset) return false;
    if (a->op != LEAF) {
      return (equivalent(a->a, b->a) && equivalent(a->b, b->b) ||
              equivalent(a->b, b->a) && equivalent(a->a, b->b));
    }
    return true;
  }

  constexpr static bool less(ScalarTreeNode* a, ScalarTreeNode* b) {
    if (a->op < b->op) return true;
    if (b->op < a->op) return false;
    if (a->is_immediate() && not b->is_immediate()) return true;
    if (b->is_immediate() && not a->is_immediate()) return false;
    if (a->constant && not b->constant) return true;
    if (b->constant && not a->constant) return false;
    if (a->offset < b->offset) return true;
    if (b->offset < a->offset) return false;
    if (a->d * to_double(a->q) < b->d * to_double(b->q)) return true;
    if (b->d * to_double(b->q) < a->d * to_double(a->q)) return false;
    return false;
  }

  // algorithms for generating simplified trees
  constexpr static ScalarTreeNode* join(Tag tag, ScalarTreeNode* a, ScalarTreeNode* b) {
    return join(tag_to_op(tag), a, b);
  }

  constexpr static ScalarTreeNode* join(Op op, ScalarTreeNode* a, ScalarTreeNode* b) {
    assert(op != LEAF && a && b);
    if (op == ADD) {
      if (a->is_zero()) {
        delete a;
        return b;
      }
      if (b->is_zero()) {
        delete b;
        return a;
      }
      if (equivalent(a, b)) {
        delete a;
        return join(MULTIPLY, new ScalarTreeNode(2), b);
      }
      return new ScalarTreeNode(ADD, a, b);
    }

    if (op == SUBTRACT) {
      if (a->is_zero()) {
        delete a;
        return join(MULTIPLY, new ScalarTreeNode(-1), b);
      }
      if (b->is_zero()) {
        delete b;
        return a;
      }
      if (equivalent(a, b)) {
        delete a;
        delete b;
        return new ScalarTreeNode(0);
      }
      return new ScalarTreeNode(SUBTRACT, a, b);
    }

    if (op == MULTIPLY) {
      if (a->is_zero()) {
        delete b;
        return a;
      }
      if (b->is_zero()) {
        delete a;
        return b;
      }
      if (a->is_one()) {
        delete a;
        return b;
      }
      if (b->is_one()) {
        delete b;
        return a;
      }
      if (a->is_immediate()) {
        b->d = a->d * b->d;
        b->q = a->q * b->q;
        delete a;
        return b;
      }
      if (b->is_immediate()) {
        a->d *= a->d;
        a->q *= a->q;
        delete b;
        return a;
      }
      ScalarTreeNode* c = new ScalarTreeNode(MULTIPLY, a, b);
      c->d = a->d * b->d;
      c->q = a->q * b->q;

      a->d = b->d = 1.0;
      a->q = b->q = Rational(1);
      return c;
    }

    if (op == DIVIDE) {
      if (a->is_zero()) {
        delete b;
        return a;
      }
      if (b->is_zero()) {
        assert(false);
      }
      if (b->is_one()) {
        delete b;
        return a;
      }
      if (b->is_immediate()) {
        a->d = a->d / b->d;
        a->q = a->q / b->q;
        delete b;
        return a;
      }
      if (equivalent(a, b)) {
        delete a;
        delete b;
        return new ScalarTreeNode(1);
      }
      ScalarTreeNode* c = new ScalarTreeNode(DIVIDE, a, b);
      c->d = a->d / b->d;
      c->q = a->q / b->q;
      a->d = b->d = 1.0;
      a->q = b->q = Rational(1);
      return c;
    }
    __builtin_unreachable();
  }

  constexpr bool is_immediate() const {
    return op == LEAF && offset < 0;
  }

  constexpr bool is_zero() const {
    return is_immediate() && (d == 0 || q == 0);
  }

  constexpr bool is_one() const {
    return is_immediate() && (d == 1 && q == 1);
  }

  constexpr int size() const {
    int n = 0;
    if (d != 1.0) ++n;
    if (q != Rational(1)) ++n;
    if (offset >= 0) ++n;
    if (op != LEAF) ++n;
    if (a) n += a->size();
    if (b) n += b->size();
    return n;
  }
};

template <typename Tree, typename Scalars, typename Constants>
struct ScalarTreeBuilder {
  const Tree& tree;
  const Scalars& scalars;
  const Constants& constants;
  int N;

  using ScalarIndex = ce::dvector<int>;

  constexpr ScalarTreeBuilder(int N, const Tree& tree, const Scalars& scalars,
                              const Constants& constants)
      : tree(tree)
      , scalars(scalars)
      , constants(constants)
      , N(N)
  {}

  // outer dimension tree generation
  constexpr void operator()(ce::dvector<const ScalarTreeNode*>& out) const
  {
    int order = tree.order();
    ScalarIndex index(order);
    do {
      out.push_back(handle(tree.root(), index));
    } while (utils::carry_sum_inc(N, 0, order, index));
  }

  // generic node dispatch
  constexpr ScalarTreeNode*
  handle(const TensorTreeNode& node, const ScalarIndex& index) const
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
     case DOUBLE:     return new ScalarTreeNode(node.d());
     case RATIONAL:   return new ScalarTreeNode(node.q());
      // case INDEX: right child of bind and partial isn't traversed
     default:         assert(false);
    }
  }

  constexpr ScalarTreeNode*
  sum(const TensorTreeNode& node, const ScalarIndex& index) const
  {
    const TensorTreeNode& a = tree.a(node);
    const TensorTreeNode& b = tree.b(node);
    Index outer = node.outer();
    ScalarTreeNode* l = handle(a, select(index, outer, a.outer()));
    ScalarTreeNode* r = handle(b, select(index, outer, b.outer()));
    return ScalarTreeNode::join(node.tag, l, r);
  }

  constexpr ScalarTreeNode*
  product(const TensorTreeNode& node, const ScalarIndex& index) const
  {
    const TensorTreeNode& a = tree.a(node);
    const TensorTreeNode& b = tree.b(node);

    if (node.tag == RATIO) {
      assert(b.order() == 0); // can't handle arbitrary inverse right now
    }

    Index outer = node.outer();
    Index inner = a.outer() & b.outer();
    Index   all = outer + inner;

    return contract(all, index, [&] (const ScalarIndex& inner) -> ScalarTreeNode* {
      ScalarTreeNode* l = handle(a, select(inner, all, a.outer()));
      ScalarTreeNode* r = handle(b, select(inner, all, b.outer()));
      return ScalarTreeNode::join(node.tag, l, r);
    });
  }

  constexpr ScalarTreeNode*
  partial(const TensorTreeNode& node, const ScalarIndex& index) const
  {
    const TensorTreeNode& a = tree.a(node);
    const TensorTreeNode& b = tree.b(node);
    assert(a.tag == TENSOR);
    assert(b.tag == INDEX);

    Index outer = node.outer();
    Index inner = b.outer();
    Index   all = outer + repeated(inner);

    return contract(all, index, [&] (const ScalarIndex& inner) -> ScalarTreeNode* {
      return tensor(a, select(inner, all, b.outer()));
    });
  }

  constexpr ScalarTreeNode*
  delta(const ScalarIndex& index) const
  {
    if (int e = index.size()) {
      int n = index[0];
      for (int i = 1; i < e; ++i) {
        if (index[i] != n) {
          return new ScalarTreeNode(0);
        }
      }
    }
    return new ScalarTreeNode(1);
  }

  constexpr ScalarTreeNode*
  tensor(const TensorTreeNode& node, const ScalarIndex& index) const
  {
    const Tensor* t = node.tensor();
    assert(t);
    if (auto&& i = constants.find(t, index)) {
      return new ScalarTreeNode(true, *i);
    }
    return new ScalarTreeNode(false, *scalars.find(t, index));
  }

  constexpr static ScalarIndex
  select(const ScalarIndex& index, const Index& from, const Index& to)
  {
    assert(index.size() == from.size());
    ScalarIndex out;
    out.reserve(to.size());
    for (char c : to) {
      out.push_back(index[*from.index_of(c)]);
    }
    return out;
  }

  template <typename Op>
  constexpr ScalarTreeNode*
  contract(const Index& all, const ScalarIndex& index, Op&& op) const
  {
    int n = index.size();
    int order = all.size();

    ScalarIndex inner(order);
    for (int i = 0; i < n; ++i) {
      inner[i] = index[i];
    }

    // return a sum of ops
    ScalarTreeNode* out = new ScalarTreeNode(0);
    do {
      out = ScalarTreeNode::join(ScalarTreeNode::ADD, out, op(inner));
    } while (utils::carry_sum_inc(N, n, order, inner));
    return out;
  }
};

struct ScalarTree
{
  ce::dvector<const ScalarTreeNode*> roots;

  template <typename Tree, typename Scalars, typename Constants>
  constexpr ScalarTree(int N, const Tree& tree, const Scalars& scalars,
                       const Constants& constants)
  {
    ScalarTreeBuilder(N, tree, scalars, constants)(roots);
  }
};
}

template <>
struct fmt::formatter<ttl::ScalarTreeNode> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::ScalarTreeNode& node, FormatContext& ctx)
  {
    auto out = format_to(ctx.out(), "{},{}", node.d, node.q);

    if (node.is_immediate()) {
      return out;
    }

    if (node.op != ttl::ScalarTreeNode::LEAF) {
      return format_to(out, " {}", ttl::ScalarTreeNode::op_to_string[node.op]);
    }

    assert(node.offset >= 0);
    const char* cstr = (node.constant) ? "constant" : "scalar";
    return format_to(out, " {}({})", cstr, node.offset);
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
    auto out = ctx.out();
    int i = 0;
    auto op = [&](const ttl::ScalarTreeNode* node, auto&& self) -> int {
      if (node->op != ttl::ScalarTreeNode::LEAF) {
        int a = self(node->a, self);
        int b = self(node->b, self);
        format_to(out, "\tnode{}[label=\"{}\"]\n", i, *node);
        format_to(out, "\tnode{} -- node{}\n", i, b);
        format_to(out, "\tnode{} -- node{}\n", i, a);
      }
      else {
        format_to(out, "\tnode{}[label=\"{}\"]\n", i, *node);
      }
      return i++;
    };

    for (const ttl::ScalarTreeNode* root : tree.roots) {
      op(root, op);
    }
    return out;
  }
};
