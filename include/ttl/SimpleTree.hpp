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
    Node*       a_ = nullptr;
    Node*       b_ = nullptr;

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
    constexpr Node(Tag tag, Node* a, Node* b)
        : tag(tag)
        , data(ttl::outer(tag, a->outer(), b->outer()))
        , constant_(a->constant_ && b->constant_)
        , a_(a)
        , b_(b)
    {
      assert(binary(tag));
    }

    // deep copy
    constexpr friend Node* clone(const Node* n) {
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

    constexpr bool constant() const {
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

  Node* root = nullptr;

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
    utils::stack<Node*> stack;
    for (auto&& node : tree) {
      Node* c = nullptr;

      if (node.tag == PARTIAL) {
        Node* b = stack.pop();
        Node* a = stack.pop();
        c = dx(a, b->index());
        delete a;
        delete b;
      }
      else if (binary(node.tag)) {
        Node* b = stack.pop();
        Node* a = stack.pop();
        c = join(node.tag, a, b);
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
  constexpr static Node* dx(const Node* n, Index index) {
    if (n == nullptr) {
      return nullptr;
    }

    if (n->constant()) {
      return new Node(0);
    }

    switch (n->tag) {
     case SUM:
     case DIFFERENCE: return distribute(n, index);
     case PRODUCT:    return product(n, index);
     case RATIO:      return quotient(n, index);
     case PARTIAL:    return partial(n, index);
     default:         return join(PARTIAL, clone(n), new Node(index));
    }
  }

  // join two subtrees, performing some basic algebra along the way
  constexpr static Node* join(Tag tag, Node* a, Node* b) {
    assert(ttl::binary(tag));
    switch (tag) {
     case SUM:
      if (a->is_zero()) return (delete a, b);
      if (b->is_zero()) return (delete b, a);
      break;
     case DIFFERENCE:
      if (a->is_zero()) return (delete a, join(PRODUCT, new Node(-1), b));
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

  constexpr static Node* partial(const Node* n, Index index) {
    return join(PARTIAL, clone(n->a()), new Node(Index(n->b()->index(), index)));
  }

  constexpr static Node* distribute(const Node* n, Index index) {
    return join(n->tag, dx(n->a(), index), dx(n->b(), index));
  }

  constexpr static Node* quotient(const Node* n, Index index) {
    Node*  p1 = join(PRODUCT, dx(n->a(), index), clone(n->b()));
    Node*  p2 = join(PRODUCT, clone(n->a()), dx(n->b(), index));
    Node* num = join(DIFFERENCE, p1, p2);
    Node* den = join(PRODUCT, clone(n->b()), clone(n->b()));
    return join(RATIO, num, den);
  }

  constexpr static Node* product(const Node* n, Index index) {
    Node* p1 = join(PRODUCT, dx(n->a(), index), clone(n->b()));
    Node* p2 = join(PRODUCT, clone(n->a()), dx(n->b(), index));
    return join(SUM, p1, p2);
  }
};
}
