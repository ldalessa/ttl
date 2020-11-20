#pragma once

#include "Scalar.hpp"
#include "ScalarIndex.hpp"
#include "Tag.hpp"
#include "TensorTree.hpp"
#include "set.hpp"
#include <memory>

namespace ttl
{
struct ScalarTree
{
  struct Node {
    Tag tag;
    Node* a_ = nullptr;
    Node* b_ = nullptr;
    bool constant = true;
    ScalarIndex index = {};

    union {
      double      d = 0;
      Rational    q;
      Tensor tensor;
    };

    constexpr ~Node() {
      assert(a_ != this);
      assert(b_ != this);
      delete a_;
      delete b_;
    }

    constexpr Node(const TensorTree::Node* tree, const ScalarIndex& i)
        : tag(TENSOR)
        , constant(tree->constant)
        , tensor(tree->tensor)
        , index(i)
    {
      assert(tree->tag == TENSOR);
    }

    constexpr Node(const Rational& q)
        : tag(RATIONAL)
        , q(q)
    {
    }

    constexpr Node(std::signed_integral auto i)
        : tag(RATIONAL)
        , q(i)
    {
    }

    constexpr Node(std::floating_point auto d)
        : tag(DOUBLE)
        , d(d)
    {
    }

    constexpr Node(Tag tag, Node* a, Node* b)
        : tag(tag)
        , a_(a)
        , b_(b)
        , constant(a->constant && b->constant)
    {
      assert(tag_is_binary(tag));
    }

    constexpr std::array<int, 2> size() const {
      if (tag_is_binary(tag)) {
        auto [a_size, a_depth] = a_->size();
        auto [b_size, b_depth] = b_->size();

        return std::array{a_size + b_size + 1, std::max(a_depth, b_depth) + 1};
      }
      return std::array{1, 1};
    }

    constexpr const ScalarIndex& outer() const {
      return index;
    }

    constexpr const Node* a() const {
      return a_;
    }

    constexpr const Node* b() const {
      return b_;
    }

    constexpr bool is_zero() const {
      return (tag == RATIONAL && q == Rational(0)) || (tag == DOUBLE && d == 0.0);
    }

    constexpr bool is_one() const {
      return (tag == RATIONAL && q == Rational(1)) || (tag == DOUBLE && d == 1.0);
    }

    constexpr void scalars(int N, set<Scalar>& out) const
    {
      if (tag_is_binary(tag)) {
        a_->scalars(N, out);
        b_->scalars(N, out);
      }
      if (tag == TENSOR) {
        out.emplace(N, this);
      }
    }

    constexpr friend bool is_equivalent(const Node* a, const Node* b) {
      assert(a && b);
      if (a->tag != b->tag) return false;
      if (a->index.size() != b->index.size()) return false;
      for (int i = 0, e = a->index.size(); i < e; ++i) {
        if (a->index[i] != b->index[i]) return false;
      }
      if (a->constant != b->constant) return false;
      switch (a->tag) { // todo: commutative
       case SUM:
       case DIFFERENCE:
       case PRODUCT:
       case RATIO: return (is_equivalent(a->a(), b->a()) && is_equivalent(a->b(), b->b()));
       case DOUBLE:   return (a->d != b->d);
       case RATIONAL: return (a->q != b->q);
       case TENSOR:   return (a->tensor != b->tensor);
       default: return true;
      }
    }

    std::string to_string(int N = 0) const
    {
      switch (tag)
      {
       case SUM:
       case DIFFERENCE:
       case PRODUCT:
       case RATIO:
        return fmt::format("({} {} {})", a_->to_string(N), tag, b_->to_string(N));

       case RATIONAL:
        return fmt::format("{}", q);

       case DOUBLE:
        return fmt::format("{}", d);

       case TENSOR:
        if (N) {
          return Scalar(N, tensor, index, constant).to_string();
        }
        if (index.size()) {
          return fmt::format("{}({})", tensor.id(), index);
        }
        return fmt::format("{}", tensor.id());
       default: assert(false);
      }
      __builtin_unreachable();
    }
  };

  int N;
  Scalar lhs_;
  Node* root_;

  constexpr ~ScalarTree() {
    delete root_;
  }

  constexpr ScalarTree(int N, const TensorTree& tree, const ScalarIndex& outer)
      : N(N)
      , lhs_(N, tree.lhs(), outer, false)
      , root_(map(tree.root(), outer))
  {
  }

  constexpr ScalarTree(const ScalarTree&) = delete;
  constexpr ScalarTree(ScalarTree&& b)
      : N(std::exchange(b.N, 0))
      , lhs_(b.lhs_)
      , root_(std::exchange(b.root_, nullptr))
  {
  }

  constexpr ScalarTree& operator=(const ScalarTree&) = delete;
  constexpr ScalarTree& operator=(ScalarTree&& b) {
    delete root_;
    N = std::exchange(b.N, 0);
    lhs_ = b.lhs_;
    root_ = std::exchange(b.root_, nullptr);
    return *this;
  }

  constexpr friend void swap(ScalarTree& a, ScalarTree& b) {
    std::swap(a.N, b.N);
    Scalar temp = a.lhs_;
    a.lhs_ = b.lhs_;
    b.lhs_ = temp;
    std::swap(a.root_, b.root_);
  }

  constexpr const Scalar& lhs() const {
    return lhs_;
  }

  constexpr const Node* root() const {
    return root_;
  }

  constexpr void scalars(set<Scalar>& out) const {
    assert(N != 0);
    assert(root_ != nullptr);
    out.emplace(lhs_);
    return root_->scalars(N, out);
  }

  constexpr std::array<int, 2> size() const {
    return root_->size();
  }

  std::string to_string() const {
    return lhs_.to_string().append(" = ").append(root_->to_string(N));
  }

 private:
  constexpr Node*
  map(const TensorTree::Node* tree, const ScalarIndex& index) const
  {
    switch (tree->tag) {
     case SUM:
     case DIFFERENCE: return sum(tree, index);
     case PRODUCT:
     case RATIO: return contract(tree, index);
     case INDEX: return delta(index);
     case TENSOR: return tensor(tree, index);
     case RATIONAL: return new Node(tree->q);
     case DOUBLE: return new Node(tree->d);
     default: assert(false);
    }
    __builtin_unreachable();
  }

  constexpr Node*
  delta(const ScalarIndex& index) const {
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
  sum(const TensorTree::Node* tree, const ScalarIndex& index) const {
    const TensorTree::Node* a = tree->a();
    const TensorTree::Node* b = tree->b();
    const Index& outer = tree->outer();
    Node* l = map(a, index.select(outer, a->outer()));
    Node* r = map(b, index.select(outer, b->outer()));
    return reduce(tree->tag, l, r);
  }

  constexpr Node*
  contract(const TensorTree::Node* tree, ScalarIndex index) const
  {
    const TensorTree::Node* a = tree->a();
    const TensorTree::Node* b = tree->b();

    // can't handle arbitrary inverse b_ now
    if (tree->tag == RATIO) {
      assert(b->order() == 0);
    }

    // Figure out what the tensor contraction looks like, e.g., matrix multiply
    // might look like `a(ij) * b(jk)` which means the outer index is `ik` and
    // we need to iterate over the `j` here.
    Index outer = tree->outer();
    Index inner = outer + (a->outer() & b->outer());

    int     n = index.size();
    int order = inner.size();

    // extend the index so we have enough space for the contracted index
    // ScalarIndex i(index);
    index.resize(order);

    // build the sum of products
    Node* out = new Node(0);
    do {
      Node*  l = map(a, index.select(inner, a->outer()));
      Node*  r = map(b, index.select(inner, b->outer()));
      Node* lr = reduce(tree->tag, l, r);
      out = reduce(SUM, out, lr);
    } while (index.carry_sum_inc(N, n));
    return out;
  }

  constexpr Node*
  tensor(const TensorTree::Node* tree, ScalarIndex index) const
  {
    // tensor indices can designate "self" contractions, like a trace `a(ii)`
    Index outer = exclusive(tree->index);
    Index inner = outer + repeated(tree->index);

    int     n = index.size();
    int order = inner.size();

    // extend the index so we have enough space for the contracted index
    // ScalarIndex i(index);
    index.resize(order);

    // build the sum of products
    Node* out = new Node(0);
    do {
      Node* t = new Node(tree, index.select(inner, tree->index));
      out = reduce(SUM, out, t);
    } while (index.carry_sum_inc(N, n));
    return out;
  }

  constexpr Node* reduce(Tag tag, Node* a, Node* b) const
  {
    // simple constant folding for rational and doubles
    if (a->tag == RATIONAL && b->tag == RATIONAL) {
      Node* out = new Node(tag_apply(tag, a->q, b->q));
      delete a;
      delete b;
      return out;
    }

    if (a->tag == DOUBLE && b->tag == DOUBLE) {
      Node* out = new Node(tag_apply(tag, a->d, b->d));
      delete a;
      delete b;
      return out;
    }

    switch (tag) {
     case SUM:        return reduce_sum(a, b);
     case DIFFERENCE: return reduce_difference(a, b);
     case PRODUCT:    return reduce_product(a, b);
     case RATIO:      return reduce_ratio(a, b);
     default: assert(false);
    }
    __builtin_unreachable();
  }

  constexpr Node* reduce_sum(Node* a, Node* b) const {
    if (a->is_zero()) {
      delete a;
      return b;
    }
    if (b->is_zero()) {
      delete b;
      return a;
    }
    if (is_equivalent(a, b)) {
      delete a;
      return new Node(PRODUCT, new Node(2), b);
    }
    return new Node(SUM, a, b);
  }

  constexpr Node* reduce_difference(Node* a, Node* b) const {
    if (b->is_zero()) {
      delete b;
      return a;
    }
    if (a->is_zero()) {
      delete a;
      return new Node(PRODUCT, new Node(-1), b);
    }
    if (is_equivalent(a, b)) {
      delete a;
      delete b;
      return new Node(0);
    }
    return new Node(DIFFERENCE, a, b);
  }

  constexpr Node* reduce_product(Node* a, Node* b) const {
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

    // canonical trees have rational numbers as left children
    if (b->tag == RATIONAL) {
      return reduce(PRODUCT, b, a);
    }

    // factor out and combine any common rational state between the two terms
    if (a->tag == PRODUCT && b->tag == PRODUCT) {
      Node* aa = a->a_;
      Node* ab = a->b_;
      Node* ba = b->a_;
      Node* bb = b->b_;
      if (aa->tag == RATIONAL && ba->tag == RATIONAL) {
        a->a_ = nullptr;
        a->b_ = nullptr;
        delete a;
        b->a_ = ab;
        b->b_ = bb;
        a = reduce(PRODUCT, aa, ba);
        return reduce(PRODUCT, a, b);
      }
      if (ba->tag == RATIONAL) {
        std::swap(b->a_, a);
        return reduce(PRODUCT, a, b);
      }
      if (aa->tag == RATIONAL) {
        a->a_ = ab;
        a->b_ = b;
        a = aa;
        return reduce(PRODUCT, a, b);
      }
    }

    return new Node(PRODUCT, a, b);
  }

  constexpr Node* reduce_ratio(Node* a, Node* b) const {
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

    // todo: not really safe if fields can have singularities
    if (is_equivalent(a, b)) {
      delete a;
      delete b;
      return new Node(1);
    }

    if (b->tag == RATIO) {
      Node* ba = std::exchange(b->a_, nullptr);
      Node* bb = std::exchange(b->b_, nullptr);
      delete b;
      return reduce(PRODUCT, a, b);
    }

    return new Node(RATIO, a, b);
  }
};

struct ScalarTreeBuilder
{
  int N;

  constexpr ScalarTreeBuilder(int N) : N(N) {}

  constexpr void
  operator()(const TensorTree& tree, ce::dvector<ScalarTree>& out) const
  {
    int order = tree.order();
    ScalarIndex index(order);
    do {
      out.emplace_back(N, tree, index);
    } while (index.carry_sum_inc(N, 0));
  }
};
}
