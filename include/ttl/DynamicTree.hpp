#pragma once

#include "concepts.hpp"
#include "TaggedNode.hpp"
#include <ce/dvector.hpp>

namespace ttl {
struct TreeNode
{
 private:
  Tag          tag_;
  Node        data_;
  bool    constant_ = true;
  TreeNode*      a_ = nullptr;
  TreeNode*      b_ = nullptr;
  TreeNode* parent_ = nullptr;

  // deep copy is private, use clone
  constexpr TreeNode(const TreeNode& b)
      : tag_(b.tag_)
      , data_(b.data_)
      , constant_(b.constant_)
      , a_(clone(b.a_))
      , b_(clone(b.b_))
  {
    if (a_) a_->parent_ = this;
    if (b_) b_->parent_ = this;
  }

 public:
  constexpr ~TreeNode() {
    delete a_;
    delete b_;
  }

  // tensor leaf
  constexpr TreeNode(Tag tag, const Node& data, bool constant)
      :      tag_(tag)
      ,     data_(data)
      , constant_(constant)
  {
    assert(tag == TENSOR);
  }

  // leaf
  constexpr TreeNode(Tag tag, const Node& data)
      :  tag_(tag)
      , data_(data)
  {
  }

  // immediate rational
  constexpr TreeNode(int n)
      :  tag_(RATIONAL)
      , data_(Rational(n))
  {
  }

  // immediate index
  constexpr TreeNode(Index index)
      : tag_(INDEX)
      , data_(index)
  {
  }

  // binary join
  constexpr TreeNode(Tag tag, TreeNode* a, TreeNode* b)
      : tag_(tag)
      , data_(ttl::outer(tag, a->outer(), b->outer()))
      , constant_(a->constant_ && b->constant_)
      , a_(a)
      , b_(b)
  {
    a_->parent_ = this;
    b_->parent_ = this;
  }

  // deep copy
  constexpr friend TreeNode* clone(const TreeNode* n) {
    return (n) ? new TreeNode(*n) : nullptr;
  }

  constexpr Tag tag() const {
    return tag_;
  }

  constexpr Index index() const {
    assert(tag_ == INDEX || tag_ == DELTA);
    return data_.index;
  }

  constexpr Tensor tensor() const {
    assert(tag_ == TENSOR);
    return data_.tensor;
  }

  constexpr Rational q() const {
    assert(tag_ == RATIONAL);
    return data_.q;
  }

  constexpr double d() const {
    assert(tag_ == DOUBLE);
    return data_.d;
  }

  constexpr bool constant() const {
    return constant_;
  }

  // left child
  constexpr const TreeNode* a() const {
    return a_;
  }

  // right child
  constexpr const TreeNode* b() const {
    return b_;
  }

  // subtree outer index
  constexpr Index outer() const {
    if (tag_ < TENSOR) {
      return data_.index;
    }
    return {};
  }

  constexpr bool is_binary() const {
    return ttl::is_binary(tag_);
  }

  constexpr bool is_zero() const {
    return tag_ == RATIONAL && q() == Rational(0);
  }

  constexpr friend int size(const TreeNode* tree) {
    return (tree) ? size(tree->a()) + size(tree->b()) + 1 : 0;
  }
};

struct DynamicTree
{
  TreeNode* root = nullptr;

  constexpr ~DynamicTree() {
    delete root;
  }

  constexpr DynamicTree(const DynamicTree&) = delete;

  constexpr DynamicTree(DynamicTree&& rhs)
      : root(std::exchange(rhs.root, nullptr))
  {
  }

  constexpr DynamicTree(is_tree auto const& tree, auto const& constants)
  {
    utils::stack<TreeNode*> stack;
    for (int i = 0; i < tree.size(); ++i) {
      Tag     tag = tree.tag(i);
      TreeNode* c = nullptr;

      if (tag == PARTIAL) {
        TreeNode* b = stack.pop();
        TreeNode* a = stack.pop();
        c = dx(a, b->index());
        delete a;
        delete b;
      }
      else if (is_binary(tag)) {
        TreeNode* b = stack.pop();
        TreeNode* a = stack.pop();
        c = join(tag, a, b);
      }
      else if (tag == TENSOR) {
        Node     node = tree.node(i);
        bool constant = constants.contains(node.tensor);
        c = new TreeNode(tag, node, constant);
      }
      else {
        c = new TreeNode(tag, tree.node(i));
      }
      stack.push(c);
    }
    root = stack.pop();
    assert(stack.size() == 0);
  }

  constexpr DynamicTree& operator=(const DynamicTree&) = delete;
  constexpr DynamicTree& operator=(DynamicTree&& rhs) {
    root = std::exchange(rhs.root, (delete root, nullptr));
    return *this;
  }

  constexpr friend int size(const DynamicTree& tree) {
    return size(tree.root);
  }

 private:
  // compute the derivative of a tree (possibly recursively)
  constexpr static TreeNode* dx(const TreeNode* n, Index index) {
    if (n == nullptr) {
      return nullptr;
    }

    if (n->constant()) {
      return new TreeNode(0);
    }

    switch (n->tag()) {
     case SUM:
     case DIFFERENCE: return distribute(n, index);
     case PRODUCT:    return product(n, index);
     case INVERSE:    return quotient(n, index);
     case PARTIAL:    return partial(n, index);
     default:         return join(PARTIAL, clone(n), new TreeNode(index));
    }
  }

  // join two subtrees, performing some basic algebra along the way
  constexpr static TreeNode* join(Tag tag, TreeNode* a, TreeNode* b) {
    assert(is_binary(tag));
    switch (tag) {
     case SUM:
      if (a->is_zero()) return (delete a, b);
      if (b->is_zero()) return (delete b, a);
      break;
     case DIFFERENCE:
      if (a->is_zero()) return (delete a, join(PRODUCT, new TreeNode(-1), b));
      if (b->is_zero()) return (delete b, a);
      break;
     case PRODUCT:
      if (a->is_zero()) return (delete b, a);
      if (b->is_zero()) return (delete a, b);
      break;
     case INVERSE:
      if (a->is_zero()) return (delete b, a);
      if (b->is_zero()) abort();
      break;
     default:
      break;
    }
    return new TreeNode(tag, a, b);
  }

  constexpr static TreeNode* partial(const TreeNode* n, Index index) {
    return join(PARTIAL, clone(n->a()), new TreeNode(n->b()->index() + index));
  }

  constexpr static TreeNode* distribute(const TreeNode* n, Index index) {
    return join(n->tag(), dx(n->a(), index), dx(n->b(), index));
  }

  constexpr static TreeNode* quotient(const TreeNode* n, Index index) {
    TreeNode*  p1 = join(PRODUCT, dx(n->a(), index), clone(n->b()));
    TreeNode*  p2 = join(PRODUCT, clone(n->a()), dx(n->b(), index));
    TreeNode* num = join(DIFFERENCE, p1, p2);
    TreeNode* den = join(PRODUCT, clone(n->b()), clone(n->b()));
    return join(INVERSE, num, den);
  }

  constexpr static TreeNode* product(const TreeNode* n, Index index) {
    TreeNode* p1 = join(PRODUCT, dx(n->a(), index), clone(n->b()));
    TreeNode* p2 = join(PRODUCT, clone(n->a()), dx(n->b(), index));
    return join(SUM, p1, p2);
  }
};
}

template <>
struct fmt::formatter<ttl::TreeNode> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::TreeNode& node, FormatContext& ctx) {
    switch (node.tag()) {
     case ttl::SUM:
     case ttl::DIFFERENCE:
     case ttl::PRODUCT:
     case ttl::INVERSE:
     case ttl::BIND:
     case ttl::PARTIAL:    return format_to(ctx.out(), "{}", node.tag());
     case ttl::INDEX:
     case ttl::DELTA:      return format_to(ctx.out(), "{}", node.index());
     case ttl::TENSOR:     return format_to(ctx.out(), "{}", node.tensor());
     case ttl::RATIONAL:   return format_to(ctx.out(), "{}", node.q());
     case ttl::DOUBLE:     return format_to(ctx.out(), "{}", node.d());
     default:
      __builtin_unreachable();
    }
  }
};

template <>
struct fmt::formatter<ttl::DynamicTree>
{
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  auto format(const ttl::DynamicTree& a, auto& ctx) {
    int i = 0;
    auto op = [&](const ttl::TreeNode* node, auto&& self) -> int {
      if (node->is_binary()) {
        int a = self(node->a(), self);
        int b = self(node->b(), self);
        format_to(ctx.out(), "\tnode{}[label=\"{} <{}>\"]\n", i, *node, node->outer());
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, a);
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, b);
      }
      else {
        format_to(ctx.out(), "\tnode{}[label=\"{}\"]\n", i, *node);
      }
      return i++;
    };
    op(a.root, op);
    return ctx.out();
  }
};
