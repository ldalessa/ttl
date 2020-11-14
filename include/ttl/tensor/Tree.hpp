#pragma once

#include "Tag.hpp"
#include "RPNTree.hpp"
#include <fmt/core.h>

namespace ttl::tensor {
struct Tree {
  Tag tag;
  Index index = {};
  Tree* a = nullptr;
  Tree* b = nullptr;
  union {
    double      d = 0;
    Rational    q;
    Tensor tensor;
  };
  bool constant = true;
  int      size = 1;

  constexpr ~Tree() {
    delete a;
    delete b;
  }

  constexpr Tree(const Tree& rhs)
      : tag(rhs.tag)
      , index(rhs.index)
      , a(clone(rhs.a))
      , b(clone(rhs.b))
      , constant(rhs.constant)
      , size(rhs.size)
  {
    if (tag == DOUBLE) d = rhs.d;
    if (tag == RATIONAL) q = rhs.q;
    if (tag == TENSOR) tensor = rhs.tensor;
  }

  constexpr Tree(const Tensor& tensor, const Index& index, bool constant)
      : tag(TENSOR)
      , index(index)
      , tensor(tensor)
      , constant(constant)
  {
    assert(tensor.order() <= index.size());
  }

  constexpr Tree(const Index& index)
      : tag(INDEX)
      , index(index)
  {
  }

  constexpr Tree(const Rational& q)
      : tag(RATIONAL)
      , q(q)
  {
  }

  constexpr Tree(std::signed_integral auto i)
      : tag(RATIONAL)
      , q(i)
  {
  }

  constexpr Tree(std::floating_point auto d)
      : tag(RATIONAL)
      , d(d)
  {
  }

  constexpr Tree(Tag tag, Tree* a, Tree* b)
      : tag(tag)
      , index(tag_outer(tag, a->outer(), b->outer()))
      , a(a)
      , b(b)
      , constant(a->constant && b->constant)
      , size(a->size + b->size + 1)
  {
    assert(tag_is_binary(tag));
  }

  constexpr friend Tree* clone(const Tree* rhs) {
    return (rhs) ? new Tree(*rhs) : nullptr;
  }

  constexpr Index outer() const {
    return exclusive(index);
  }

  constexpr bool is_zero() const {
    return tag == RATIONAL && q == Rational(0);
  }

  constexpr bool is_one() const {
    return tag == RATIONAL && q == Rational(1);
  }

  constexpr friend bool is_equivalent(const Tree* a, const Tree* b) {
    assert(a && b);
    if (a->tag != b->tag) return false;
    if (a->index != b->index) return false;
    if (a->constant != b->constant) return false;
    switch (a->tag) { // todo: commutative
     case SUM:
     case DIFFERENCE:
     case PRODUCT:
     case RATIO: return (is_equivalent(a->a, b->a) && is_equivalent(a->b, b->b));
     case DOUBLE:   return (a->d != b->d);
     case RATIONAL: return (a->q != b->q);
     case TENSOR:   return (a->tensor != b->tensor);
     default: return true;
    }
  }
};

template <typename Constants>
struct TreeBuilder {
  Constants constants;

  constexpr TreeBuilder(Constants&& constants) : constants(std::move(constants)) {}

  template <int M>
  constexpr const Tree* operator()(const RPNTree<M>& tree) const {
    return map(tree, tree.root());
  }

  template <int M>
  constexpr Tree* map(const RPNTree<M>& tree, const RPNNode* node) const {
    switch (node->tag) {
     default:
      return reduce(node->tag, map(tree, tree.a(node)), map(tree, tree.b(node)));
     case PARTIAL:
      return dx(map(tree, tree.a(node)), tree.b(node)->index);
     case INDEX:
      return new Tree(node->index);
     case TENSOR:
      return new Tree(node->tensor, node->index, constants(node->tensor));
     case RATIONAL:
      return new Tree(node->q);
     case DOUBLE:
      return new Tree(node->d);
    }
  }

  constexpr Tree* reduce(Tag tag, Tree* a, Tree* b) const {
    switch (tag) {
     case SUM:        return reduce_sum(a, b);
     case DIFFERENCE: return reduce_difference(a, b);
     case PRODUCT:    return reduce_product(a, b);
     case RATIO:      return reduce_ratio(a, b);
     default: assert(false);
    }
  }

  constexpr Tree* reduce_sum(Tree* a, Tree* b) const {
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
      return new Tree(PRODUCT, new Tree(2), b);
    }
    return new Tree(SUM, a, b);
  }

  constexpr Tree* reduce_difference(Tree* a, Tree* b) const {
    if (b->is_zero()) {
      delete b;
      return a;
    }
    if (a->is_zero()) {
      delete a;
      return new Tree(PRODUCT, new Tree(-1), b);
    }
    if (is_equivalent(a, b)) {
      delete a;
      delete b;
      return new Tree(0);
    }
    return new Tree(DIFFERENCE, a, b);
  }

  constexpr Tree* reduce_product(Tree* a, Tree* b) const {
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
    return new Tree(PRODUCT, a, b);
  }

  constexpr Tree* reduce_ratio(Tree* a, Tree* b) const {
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
      return new Tree(1);
    }
    return new Tree(RATIO, a, b);
  }

  constexpr Tree* dx(Tree* node, const Index& index) const {
    if (node->constant) {
      delete node;
      return new Tree(0);
    }

    if (node->tag == TENSOR) {
      node->index += index;
      return node;
    }

    Tag tag = node->tag;
    Tree* a = std::exchange(node->a, nullptr);
    Tree* b = std::exchange(node->b, nullptr);
    delete node;

    switch (tag) {
     case SUM:        return reduce(SUM, dx(a, index), dx(b, index));
     case DIFFERENCE: return reduce(DIFFERENCE, dx(a, index), dx(b, index));
     case PRODUCT:    return dx_product(a, b, index);
     case RATIO:      return dx_quotient(a, b, index);
     default: assert(false);
    }
  }

  constexpr Tree* dx_product(Tree* a, Tree* b, const Index& index) const {
    if (a->constant) {
      return reduce(PRODUCT, a, dx(b, index));
    }
    if (b->constant) {
      return reduce(PRODUCT, dx(a, index), b);
    }

    // (a'b + ab')
    Tree* t = reduce(PRODUCT, dx(clone(a), index), clone(b));
    Tree* u = reduce(PRODUCT, a, dx(b, index));
    return reduce(SUM, t, u);
  }

  constexpr Tree* dx_quotient(Tree* a, Tree* b, const Index& index) const {
    if (b->constant) {
      return reduce(RATIO, dx(a, index), b);
    }

    // (a'b - ab')/b^2
    Tree*   b2 = reduce(PRODUCT, clone(b), clone(b));
    Tree* ap_b = reduce(PRODUCT, dx(clone(a), index), clone(b));
    Tree* a_bp = reduce(PRODUCT, a, dx(b, index));
    return reduce(RATIO, reduce(DIFFERENCE, ap_b, a_bp), b2);
  }
};
}

template <>
struct fmt::formatter<ttl::tensor::Tree>
{
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::tensor::Tree& tree, FormatContext& ctx)
  {
    using namespace ttl::tensor;
    auto op = [&](const Tree& tree, auto&& self) -> std::string {
      switch (tree.tag) {
       case SUM:
       case DIFFERENCE:
       case PRODUCT:
       case RATIO: {
        std::string b = self(*tree.b, self);
        std::string a = self(*tree.a, self);
        return fmt::format("({} {} {})", a, tree.tag, b);
       }
       case INDEX:    return fmt::format("{}", tree.index);
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
