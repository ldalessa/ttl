#pragma once

#include "ParseTree.hpp"
#include <memory>

namespace ttl {
struct TensorTree
{
  struct Node {
    Tag tag;
    Index index = {};
    Node* a_ = nullptr;
    Node* b_ = nullptr;
    union {
      double      d;
      Rational    q;
      Tensor tensor;
    };
    bool constant = true;
    int      size = 1;

    constexpr ~Node() {
      delete a_;
      delete b_;
    }

    constexpr Node(const Node& rhs)
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

    constexpr Node(Node&& rhs)
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

    constexpr Node(const Tensor& tensor, const Index& index, bool constant)
        : tag(TENSOR)
        , index(index)
        , tensor(tensor)
        , constant(constant)
    {
      assert(tensor.order() <= index.size());
    }

    constexpr Node(const Index& index)
        : tag(INDEX)
        , index(index)
    {
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
        , index(tag_outer(tag, a->outer(), b->outer()))
        , a_(a)
        , b_(b)
        , constant(a->constant && b->constant)
        , size(a->size + b->size + 1)
    {
      assert(tag_is_binary(tag));
    }

    constexpr friend Node* clone(const Node* rhs) {
      return (rhs) ? new Node(*rhs) : nullptr;
    }

    constexpr const Node* a() const {
      return a_;
    }

    constexpr const Node* b() const {
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

    constexpr friend bool is_equivalent(const Node* a, const Node* b) {
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

  Node* root_;

  constexpr ~TensorTree() {
    delete root_;
  }

  constexpr TensorTree() = default;

  template <int M>
  constexpr TensorTree(const ParseTree<M>& tree, auto&& constants)
      : root_(map(tree.root(), constants))
  {
  }

  constexpr TensorTree(const TensorTree&) = delete;
  constexpr TensorTree(TensorTree&& b)
      : root_(std::exchange(b.root_, nullptr))
  {
  }

  constexpr const Node* root() const {
    return root_;
  }

  constexpr Index outer() const {
    return root_->outer();
  }

  constexpr int order() const {
    return outer().size();
  }

  std::string to_string() const {
    return root_->to_string();
  }

 private:
  constexpr static Node* map(const ParseNode* node, auto const& constants)
  {
    switch (node->tag) {
     default:
      return reduce(node->tag, map(node->a(), constants), map(node->b(), constants));
     case PARTIAL:
      return dx(map(node->a(), constants), node->b()->index);
     case INDEX:
      return new Node(node->index);
     case TENSOR:
      return new Node(node->tensor, node->index, constants(node->tensor));
     case RATIONAL:
      return new Node(node->q);
     case DOUBLE:
      return new Node(node->d);
    }
  }

  constexpr static Node* reduce(Tag tag, Node* a, Node* b)
  {
    switch (tag) {
     case SUM:        return reduce_sum(a, b);
     case DIFFERENCE: return reduce_difference(a, b);
     case PRODUCT:    return reduce_product(a, b);
     case RATIO:      return reduce_ratio(a, b);
     default: assert(false);
    }
  }

  constexpr static Node* reduce_sum(Node* a, Node* b)
  {
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

  constexpr static Node* reduce_difference(Node* a, Node* b)
  {
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

  constexpr static Node* reduce_product(Node* a, Node* b)
  {
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
    return new Node(PRODUCT, a, b);
  }

  constexpr static Node* reduce_ratio(Node* a, Node* b)
  {
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
      return new Node(1);
    }
    return new Node(RATIO, a, b);
  }

  constexpr static Node* dx(Node* node, const Index& index)
  {
    if (node->constant) {
      delete node;
      return new Node(0);
    }

    if (node->tag == TENSOR) {
      node->index += index;
      return node;
    }

    Tag tag = node->tag;
    Node* a = std::exchange(node->a_, nullptr);
    Node* b = std::exchange(node->b_, nullptr);
    delete node;

    switch (tag) {
     case SUM:        return reduce(SUM, dx(a, index), dx(b, index));
     case DIFFERENCE: return reduce(DIFFERENCE, dx(a, index), dx(b, index));
     case PRODUCT:    return dx_product(a, b, index);
     case RATIO:      return dx_quotient(a, b, index);
     default: assert(false);
    }
  }

  constexpr static Node* dx_product(Node* a, Node* b, const Index& index)
  {
    if (a->constant) {
      return reduce(PRODUCT, a, dx(b, index));
    }
    if (b->constant) {
      return reduce(PRODUCT, dx(a, index), b);
    }

    // (a'b + ab')
    Node* t = reduce(PRODUCT, dx(clone(a), index), clone(b));
    Node* u = reduce(PRODUCT, a, dx(b, index));
    return reduce(SUM, t, u);
  }

  constexpr static Node* dx_quotient(Node* a, Node* b, const Index& index)
  {
    if (b->constant) {
      return reduce(RATIO, dx(a, index), b);
    }

    // (a'b - ab')/b^2
    Node*   b2 = reduce(PRODUCT, clone(b), clone(b));
    Node* ap_b = reduce(PRODUCT, dx(clone(a), index), clone(b));
    Node* a_bp = reduce(PRODUCT, a, dx(b, index));
    return reduce(RATIO, reduce(DIFFERENCE, ap_b, a_bp), b2);
  }
};
}
