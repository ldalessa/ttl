#pragma once

#include "ParseTree.hpp"
#include <memory>

namespace ttl {
struct TensorTree {
  Tag tag;
  Index index = {};
  TensorTree* a_ = nullptr;
  TensorTree* b_ = nullptr;
  union {
    double      d;
    Rational    q;
    Tensor tensor;
  };
  bool constant = true;
  int      size = 1;

  constexpr ~TensorTree() {
    delete a_;
    delete b_;
  }

  constexpr TensorTree(const TensorTree& rhs)
      : tag(rhs.tag)
      , index(rhs.index)
      , a_(clone(rhs.a_))
      , b_(clone(rhs.b_))
      , constant(rhs.constant)
      , size(rhs.size)
  {
    if (tag == DOUBLE) std::construct_at(&d, rhs.d);
    if (tag == RATIONAL) std::construct_at(&q, rhs.q);
    if (tag == TENSOR) std::construct_at(&tensor, rhs.tensor);
  }

  constexpr TensorTree(TensorTree&& rhs)
      : tag(rhs.tag)
      , index(rhs.index)
      , a_(std::exchange(rhs.a_, nullptr))
      , b_(std::exchange(rhs.b_, nullptr))
      , constant(rhs.constant)
      , size(rhs.size)
  {
    if (tag == DOUBLE) std::construct_at(&d, rhs.d);
    if (tag == RATIONAL) std::construct_at(&q, rhs.q);
    if (tag == TENSOR) std::construct_at(&tensor, rhs.tensor);
  }

  constexpr TensorTree(const Tensor& tensor, const Index& index, bool constant)
      : tag(TENSOR)
      , index(index)
      , tensor(tensor)
      , constant(constant)
  {
    assert(tensor.order() <= index.size());
  }

  constexpr TensorTree(const Index& index)
      : tag(INDEX)
      , index(index)
  {
  }

  constexpr TensorTree(const Rational& q)
      : tag(RATIONAL)
      , q(q)
  {
  }

  constexpr TensorTree(std::signed_integral auto i)
      : tag(RATIONAL)
      , q(i)
  {
  }

  constexpr TensorTree(std::floating_point auto d)
      : tag(DOUBLE)
      , d(d)
  {
  }

  constexpr TensorTree(Tag tag, TensorTree* a, TensorTree* b)
      : tag(tag)
      , index(tag_outer(tag, a->outer(), b->outer()))
      , a_(a)
      , b_(b)
      , constant(a->constant && b->constant)
      , size(a->size + b->size + 1)
  {
    assert(tag_is_binary(tag));
  }

  constexpr friend TensorTree* clone(const TensorTree* rhs) {
    return (rhs) ? new TensorTree(*rhs) : nullptr;
  }

  constexpr const TensorTree* a() const {
    return a_;
  }

  constexpr const TensorTree* b() const {
    return b_;
  }

  constexpr Index outer() const {
    return (tag == TENSOR) ? exclusive(index) : index;
  }

  constexpr int order() const {
    return outer().size();
  }

  constexpr bool is_zero() const {
    return tag == RATIONAL && q == Rational(0);
  }

  constexpr bool is_one() const {
    return tag == RATIONAL && q == Rational(1);
  }

  constexpr friend bool is_equivalent(const TensorTree* a, const TensorTree* b) {
    assert(a && b);
    if (a->tag != b->tag) return false;
    if (a->index != b->index) return false;
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

  std::string to_string() const
  {
    switch (tag)
    {
     case SUM:
     case DIFFERENCE:
     case PRODUCT:
     case RATIO:
      return fmt::format("({} {} {})", a_->to_string(), tag, b_->to_string());

     case INDEX:    return fmt::format("{}", index);
     case RATIONAL: return fmt::format("{}", q);
     case DOUBLE:   return fmt::format("{}", d);
     case TENSOR:
      if (index.size()) {
        return fmt::format("{}({})", tensor, index);
      }
      else {
        return fmt::format("{}", tensor);
      }
     default: assert(false);
    }
  }
};

template <typename Constants>
struct TensorTreeBuilder {
  Constants constants;

  constexpr TensorTreeBuilder(Constants&& constants) : constants(std::move(constants)) {}

  template <int M>
  constexpr const TensorTree* operator()(const ParseTree<M>& tree) const {
    return map(tree.root());
  }

  constexpr TensorTree* map(const ParseNode* node) const {
    switch (node->tag) {
     default:
      return reduce(node->tag, map(node->a()), map(node->b()));
     case PARTIAL:
      return dx(map(node->a()), node->b()->index);
     case INDEX:
      return new TensorTree(node->index);
     case TENSOR:
      return new TensorTree(node->tensor, node->index, constants(node->tensor));
     case RATIONAL:
      return new TensorTree(node->q);
     case DOUBLE:
      return new TensorTree(node->d);
    }
  }

  constexpr TensorTree* reduce(Tag tag, TensorTree* a, TensorTree* b) const {
    switch (tag) {
     case SUM:        return reduce_sum(a, b);
     case DIFFERENCE: return reduce_difference(a, b);
     case PRODUCT:    return reduce_product(a, b);
     case RATIO:      return reduce_ratio(a, b);
     default: assert(false);
    }
  }

  constexpr TensorTree* reduce_sum(TensorTree* a, TensorTree* b) const {
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
      return new TensorTree(PRODUCT, new TensorTree(2), b);
    }
    return new TensorTree(SUM, a, b);
  }

  constexpr TensorTree* reduce_difference(TensorTree* a, TensorTree* b) const {
    if (b->is_zero()) {
      delete b;
      return a;
    }
    if (a->is_zero()) {
      delete a;
      return new TensorTree(PRODUCT, new TensorTree(-1), b);
    }
    if (is_equivalent(a, b)) {
      delete a;
      delete b;
      return new TensorTree(0);
    }
    return new TensorTree(DIFFERENCE, a, b);
  }

  constexpr TensorTree* reduce_product(TensorTree* a, TensorTree* b) const {
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
    return new TensorTree(PRODUCT, a, b);
  }

  constexpr TensorTree* reduce_ratio(TensorTree* a, TensorTree* b) const {
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
    if (is_equivalent(a, b)) {
      // todo: not really safe
      delete a;
      delete b;
      return new TensorTree(1);
    }
    return new TensorTree(RATIO, a, b);
  }

  constexpr TensorTree* dx(TensorTree* node, const Index& index) const {
    if (node->constant) {
      delete node;
      return new TensorTree(0);
    }

    if (node->tag == TENSOR) {
      node->index += index;
      return node;
    }

    Tag tag = node->tag;
    TensorTree* a = std::exchange(node->a_, nullptr);
    TensorTree* b = std::exchange(node->b_, nullptr);
    delete node;

    switch (tag) {
     case SUM:        return reduce(SUM, dx(a, index), dx(b, index));
     case DIFFERENCE: return reduce(DIFFERENCE, dx(a, index), dx(b, index));
     case PRODUCT:    return dx_product(a, b, index);
     case RATIO:      return dx_quotient(a, b, index);
     default: assert(false);
    }
  }

  constexpr TensorTree* dx_product(TensorTree* a, TensorTree* b, const Index& index) const {
    if (a->constant) {
      return reduce(PRODUCT, a, dx(b, index));
    }
    if (b->constant) {
      return reduce(PRODUCT, dx(a, index), b);
    }

    // (a'b + ab')
    TensorTree* t = reduce(PRODUCT, dx(clone(a), index), clone(b));
    TensorTree* u = reduce(PRODUCT, a, dx(b, index));
    return reduce(SUM, t, u);
  }

  constexpr TensorTree* dx_quotient(TensorTree* a, TensorTree* b, const Index& index) const {
    if (b->constant) {
      return reduce(RATIO, dx(a, index), b);
    }

    // (a'b - ab')/b^2
    TensorTree*   b2 = reduce(PRODUCT, clone(b), clone(b));
    TensorTree* ap_b = reduce(PRODUCT, dx(clone(a), index), clone(b));
    TensorTree* a_bp = reduce(PRODUCT, a, dx(b, index));
    return reduce(RATIO, reduce(DIFFERENCE, ap_b, a_bp), b2);
  }
};

template <typename Constants>
TensorTreeBuilder(Constants) -> TensorTreeBuilder<Constants>;
}
