#pragma once

#include "Node.hpp"
#include "concepts.hpp"
#include "utils.hpp"

namespace ttl {

struct SimpleTree
{
  struct Node
  {
    Tag       tag;
    Data     data;

   private:
    bool constant_ = true;
    const Node* a_ = nullptr;
    const Node* b_ = nullptr;

    // deep copy is private, use clone
    constexpr Node(const Node& b)
        : tag(b.tag)
        , data(b.data)
        , constant_(b.constant_)
        , a_(clone(b.a_))
        , b_(clone(b.b_))
    {
    }

   public:
    constexpr ~Node() {
      delete a_;
      delete b_;
    }

    // tensor leaf
    constexpr Node(Tag tag, const Data& data, bool constant)
        :      tag(tag)
        ,     data(data)
        , constant_(constant)
    {
      assert(tag == TENSOR);
    }

    // leaf
    constexpr Node(Tag tag, const Data& data)
        :  tag(tag)
        , data(data)
    {
      assert(!binary(tag));
    }

    // immediate rational
    constexpr Node(int n)
        :  tag(RATIONAL)
        , data(Rational(n))
    {
    }

    // immediate index
    constexpr Node(Index index)
        : tag(INDEX)
        , data(index)
    {
    }

    // binary join
    constexpr Node(Tag tag, const Node* a, const Node* b)
        : tag(tag)
        , data(ttl::outer(tag, a->outer(), b->outer()))
        , constant_(a->is_constant() && b->is_constant())
        , a_(a)
        , b_(b)
    {
      assert(binary(tag));
    }

    // deep copy
    constexpr friend const Node* clone(const Node* n) {
      return (n) ? new Node(*n) : nullptr;
    }

    constexpr const Index& index() const {
      assert(tag == INDEX || tag == DELTA);
      return data.index;
    }

    constexpr const Tensor* tensor() const {
      assert(tag == TENSOR);
      return data.tensor;
    }

    constexpr const Rational& q() const {
      assert(tag == RATIONAL);
      return data.q;
    }

    constexpr double d() const {
      assert(tag == DOUBLE);
      return data.d;
    }

    constexpr bool is_constant() const {
      return constant_;
    }

    // left child
    constexpr const Node* a() const {
      return a_;
    }

    // right child
    constexpr const Node* b() const {
      return b_;
    }

    // subtree outer index
    constexpr Index outer() const {
      if (ttl::index(tag)) {
        return data.index;
      }
      return {};
    }

    constexpr bool is_binary() const {
      return ttl::binary(tag);
    }

    constexpr bool is_zero() const {
      return tag == RATIONAL && q() == Rational(0);
    }

    constexpr int size() const {
      return (ttl::leaf(tag)) ?: a_->size() + b_->size() + 1;
    }
  };

  const Node* root = nullptr;

  constexpr ~SimpleTree() {
    delete root;
  }

  constexpr SimpleTree(const SimpleTree&) = delete;

  constexpr SimpleTree(SimpleTree&& rhs)
      : root(std::exchange(rhs.root, nullptr))
  {
  }

  constexpr SimpleTree(is_tree auto const& tree, auto const& constants)
  {
    utils::stack<const Node*> stack;
    for (auto&& node : tree) {
      const Node* c = nullptr;

      if (node.tag == PARTIAL) {
        const Node* b = stack.pop();
        const Node* a = stack.pop();
        c = dx(a, b->index());
        delete a;
        delete b;
      }
      else if (binary(node.tag)) {
        const Node* b = stack.pop();
        const Node* a = stack.pop();
        c = combine(node.tag, a, b);
      }
      else if (const Tensor* t = node.tensor()) {
        c = new Node(node.tag, node.data, utils::contains(constants, t));
      }
      else {
        c = new Node(node.tag, node.data);
      }
      stack.push(c);
    }
    root = stack.pop();
    assert(stack.size() == 0);
  }

  constexpr SimpleTree& operator=(const SimpleTree&) = delete;
  constexpr SimpleTree& operator=(SimpleTree&& rhs) {
    root = std::exchange(rhs.root, (delete root, nullptr));
    return *this;
  }

  constexpr int size() const {
    return root->size();
  }

  template <typename Tree> requires(is_tree<Tree>)
  constexpr friend Tree to_tree(const SimpleTree& tree) {
    Tree out;
    int i = out.size() - 1;
    auto op = [&](const Node* node, auto&& self) -> int {
      int j = i--;
      out[j].tag = node->tag;
      out[j].data = node->data;
      int left = j;
      if (node->is_binary()) {
        self(node->b(), self);
        left = self(node->a(), self);
      }
      out[j].left = j - left;
      return j;
    };
    op(tree.root, op);
    return out;
  }

 private:
  // compute the derivative of a tree (possibly recursively)
  constexpr static const Node* dx(const Node* n, Index index) {
    if (n == nullptr) {
      return nullptr;
    }

    if (n->is_constant()) {
      return new Node(0);
    }

    switch (n->tag) {
     case SUM:
     case DIFFERENCE: return distribute(n, index);
     case PRODUCT:    return product(n, index);
     case RATIO:      return quotient(n, index);
     case BIND:
     case PARTIAL:    return partial(n, index);
     case TENSOR:     return combine(PARTIAL, clone(n), new Node(index));
     default:
      assert(false);
    }
  }

  // combine two subtrees, performing some basic algebra along the way
  constexpr static const Node*
  combine(Tag tag, const Node* a, const Node* b)
  {
    assert(ttl::binary(tag));
    switch (tag) {
     case SUM:
      if (a->is_zero()) return (delete a, b);
      if (b->is_zero()) return (delete b, a);
      break;
     case DIFFERENCE:
      if (a->is_zero()) return (delete a, combine(PRODUCT, new Node(-1), b));
      if (b->is_zero()) return (delete b, a);
      break;
     case PRODUCT:
      if (a->is_zero()) return (delete b, a);
      if (b->is_zero()) return (delete a, b);
      break;
     case RATIO:
      if (a->is_zero()) return (delete b, a);
      if (b->is_zero()) abort();
      break;
     default:
      break;
    }
    return new Node(tag, a, b);
  }

  constexpr static const Node*
  partial(const Node* node, const Index& index)
  {
    const Node* a = node->a();
    const Node* b = node->b();
    return combine(PARTIAL, clone(a), new Node(Index(b->index(), index)));
  }

  constexpr static const Node*
  distribute(const Node* node, const Index& index)
  {
    const Node* a = node->a();
    const Node* b = node->b();
    return combine(node->tag, dx(a, index), dx(b, index));
  }

  constexpr static const Node*
  quotient(const Node* node, const Index& index)
  {
    const Node* a = node->a();
    const Node* b = node->b();

    // easy and important case when b is constant
    if (b->is_constant()) {
      return combine(RATIO, dx(a, index), clone(b));
    }

    const Node*  p1 = combine(PRODUCT, dx(a, index), clone(b));
    const Node*  p2 = combine(PRODUCT, clone(a), dx(b, index));
    const Node* num = combine(DIFFERENCE, p1, p2);
    const Node* den = combine(PRODUCT, clone(b), clone(b));
    return combine(RATIO, num, den);
  }

  constexpr static const Node*
  product(const Node* node, const Index& index)
  {
    const Node* a = node->a();
    const Node* b = node->b();

    // pre-simplify constant multiplication
    if (a->is_constant()) {
      return combine(PRODUCT, clone(a), dx(b, index));
    }

    if (b->is_constant()) {
      return combine(PRODUCT, dx(a, index), clone(b));
    }

    const Node* p1 = combine(PRODUCT, dx(a, index), clone(b));
    const Node* p2 = combine(PRODUCT, clone(a), dx(b, index));
    return combine(SUM, p1, p2);
  }
};
}
