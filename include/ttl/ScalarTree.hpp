#pragma once

#include "ScalarIndex.hpp"
#include "Tag.hpp"
#include "TensorTree.hpp"
#include "utils.hpp"
#include <memory>

namespace ttl
{
struct ScalarTree
{
  Tag           tag;
  ScalarTree*    a_ = nullptr;
  ScalarTree*    b_ = nullptr;
  bool     constant = true;
  int          size = 1;
  ScalarIndex index = {};

  union {
    double      d = 0;
    Rational    q;
    Tensor tensor;
  };

  constexpr ~ScalarTree() {
    assert(a_ != this);
    assert(b_ != this);
    delete a_;
    delete b_;
  }

  constexpr ScalarTree(const TensorTree* tree, const ScalarIndex& i)
      : tag(TENSOR)
      , constant(tree->constant)
      , tensor(tree->tensor)
      , index(i)
  {
    assert(tree->tag == TENSOR);
  }

  constexpr ScalarTree(const Rational& q)
      : tag(RATIONAL)
      , q(q)
  {
  }

  constexpr ScalarTree(std::signed_integral auto i)
      : tag(RATIONAL)
      , q(i)
  {
  }

  constexpr ScalarTree(std::floating_point auto d)
      : tag(DOUBLE)
      , d(d)
  {
  }

  constexpr ScalarTree(Tag tag, ScalarTree* a, ScalarTree* b)
      : tag(tag)
      , a_(a)
      , b_(b)
      , constant(a->constant && b->constant)
  {
    assert(tag_is_binary(tag));
  }

  constexpr const ScalarIndex& outer() const {
    return index;
  }

  constexpr const ScalarTree* a() const {
    return a_;
  }

  constexpr const ScalarTree* b() const {
    return b_;
  }

  constexpr bool is_zero() const {
    return (tag == RATIONAL && q == Rational(0)) || (tag == DOUBLE && d == 0.0);
  }

  constexpr bool is_one() const {
    return (tag == RATIONAL && q == Rational(1)) || (tag == DOUBLE && d == 1.0);
  }

  constexpr friend bool is_equivalent(const ScalarTree* a, const ScalarTree* b) {
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
};

struct ScalarTreeBuilder
{
  int N;

  constexpr ScalarTreeBuilder(int N) : N(N) {}

  constexpr void
  operator()(const TensorTree* tree, ce::dvector<utils::box<const ScalarTree>>& out) const
  {
    int order = tree->order();
    ScalarIndex index(order);
    do {
      out.emplace_back(map(tree, index));
    } while (utils::carry_sum_inc(N, 0, order, index));
  }

  constexpr ScalarTree*
  map(const TensorTree* tree, const ScalarIndex& index) const
  {
    switch (tree->tag) {
     case SUM:
     case DIFFERENCE: return sum(tree, index);
     case PRODUCT:
     case RATIO: return contract(tree, index);
     case INDEX: return delta(index);
     case TENSOR: return tensor(tree, index);
     case RATIONAL: return new ScalarTree(tree->q);
     case DOUBLE: return new ScalarTree(tree->d);
     default: assert(false);
    }
  }

  constexpr ScalarTree*
  delta(const ScalarIndex& index) const {
    if (int e = index.size()) {
      int n = index[0];
      for (int i = 1; i < e; ++i) {
        if (index[i] != n) {
          return new ScalarTree(0);
        }
      }
    }
    return new ScalarTree(1);
  }

  constexpr ScalarTree*
  sum(const TensorTree* tree, const ScalarIndex& index) const {
    const TensorTree* a = tree->a();
    const TensorTree* b = tree->b();
    const Index& outer = tree->outer();
    ScalarTree* l = map(a, index.select(outer, a->outer()));
    ScalarTree* r = map(b, index.select(outer, b->outer()));
    return reduce(tree->tag, l, r);
  }

  constexpr ScalarTree*
  contract(const TensorTree* tree, ScalarIndex index) const
  {
    const TensorTree* a = tree->a();
    const TensorTree* b = tree->b();

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
    ScalarTree* out = new ScalarTree(0);
    do {
      ScalarTree*  l = map(a, index.select(inner, a->outer()));
      ScalarTree*  r = map(b, index.select(inner, b->outer()));
      ScalarTree* lr = reduce(tree->tag, l, r);
      out = reduce(SUM, out, lr);
    } while (utils::carry_sum_inc(N, n, order, index));
    return out;
  }

  constexpr ScalarTree*
  tensor(const TensorTree* tree, ScalarIndex index) const
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
    ScalarTree* out = new ScalarTree(0);
    do {
      ScalarTree* t = new ScalarTree(tree, index.select(inner, tree->index));
      out = reduce(SUM, out, t);
    } while (utils::carry_sum_inc(N, n, order, index));
    return out;
  }

  constexpr ScalarTree* reduce(Tag tag, ScalarTree* a, ScalarTree* b) const
  {
    // simple constant folding for rational and doubles
    if (a->tag == RATIONAL && b->tag == RATIONAL) {
      ScalarTree* out = new ScalarTree(tag_apply(tag, a->q, b->q));
      delete a;
      delete b;
      return out;
    }

    if (a->tag == DOUBLE && b->tag == DOUBLE) {
      ScalarTree* out = new ScalarTree(tag_apply(tag, a->d, b->d));
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
  }

  constexpr ScalarTree* reduce_sum(ScalarTree* a, ScalarTree* b) const {
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
      return new ScalarTree(PRODUCT, new ScalarTree(2), b);
    }
    return new ScalarTree(SUM, a, b);
  }

  constexpr ScalarTree* reduce_difference(ScalarTree* a, ScalarTree* b) const {
    if (b->is_zero()) {
      delete b;
      return a;
    }
    if (a->is_zero()) {
      delete a;
      return new ScalarTree(PRODUCT, new ScalarTree(-1), b);
    }
    if (is_equivalent(a, b)) {
      delete a;
      delete b;
      return new ScalarTree(0);
    }
    return new ScalarTree(DIFFERENCE, a, b);
  }

  constexpr ScalarTree* reduce_product(ScalarTree* a, ScalarTree* b) const {
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
      ScalarTree* aa = a->a_;
      ScalarTree* ab = a->b_;
      ScalarTree* ba = b->a_;
      ScalarTree* bb = b->b_;
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

    return new ScalarTree(PRODUCT, a, b);
  }

  constexpr ScalarTree* reduce_ratio(ScalarTree* a, ScalarTree* b) const {
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
      return new ScalarTree(1);
    }

    if (b->tag == RATIO) {
      ScalarTree* ba = std::exchange(b->a_, nullptr);
      ScalarTree* bb = std::exchange(b->b_, nullptr);
      delete b;
      return reduce(PRODUCT, a, b);
    }

    return new ScalarTree(RATIO, a, b);
  }
};
}

template <>
struct fmt::formatter<ttl::ScalarTree>
{
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::ScalarTree& tree, FormatContext& ctx)
  {
    using namespace ttl;
    auto op = [&](const ScalarTree& tree, auto&& self) -> std::string {
      switch (tree.tag) {
       case SUM:
       case DIFFERENCE:
       case PRODUCT:
       case RATIO: {
         std::string b = self(*tree.b(), self);
         std::string a = self(*tree.a(), self);
         return fmt::format("({} {} {})", a, tree.tag, b);
       }

       case RATIONAL: return fmt::format("{}", tree.q);
       case DOUBLE:   return fmt::format("{}", tree.d);
       case TENSOR:
        if (tree.index.size()) {
          return fmt::format("{}({})", tree.tensor, tree.index);
        }
        else {
          return fmt::format("{}", tree.tensor);
        }
       default: assert(false);
      }
    };
    return format_to(ctx.out(), "{}", op(tree, op));
  }
};
