#pragma once

#include "Index.hpp"
#include "Rational.hpp"
#include "ParseTree.hpp"
#include "Scalar.hpp"
#include "TreeShape.hpp"
#include "pow.hpp"
#include "set.hpp"
#include <memory>

namespace ttl
{
  struct TensorTree
  {
    using is_tree_tag = void;

    struct Node
    {
      Tag tag = {};
      Index index = {};
      Node* a_ = nullptr;
      Node* b_ = nullptr;
      union {
        double      d = 0;
        Rational    q;
        Tensor tensor;
      };
      bool constant = true;
      int      size = 1;

      constexpr ~Node() {
        delete a_;
        delete b_;
      }

      constexpr Node(Node const& rhs)
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

      constexpr Node(Tensor const& tensor, Index const& index, bool constant)
          : tag(TENSOR)
          , index(index)
          , tensor(tensor)
          , constant(constant)
      {
        assert(tensor.order() <= index.size());
      }

      constexpr Node(Index const& index)
          : tag(INDEX)
          , index(index)
      {
      }

      constexpr Node(Rational const& q)
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

      constexpr friend auto clone(const Node* rhs) -> Node*
      {
        return (rhs) ? new Node(*rhs) : nullptr;
      }

      constexpr auto a() const -> Node const*
      {
        return a_;
      }

      constexpr auto b() const -> Node const*
      {
        return b_;
      }

      constexpr auto outer() const -> Index
      {
        return (tag == TENSOR) ? exclusive(index) : index;
      }

      constexpr auto all() const -> Index
      {
        if (tag == TENSOR) {
          return unique(index) + repeated(index);
        }
        if (tag == PRODUCT || tag == RATIO) {
          return index + a_->outer() & b_->outer();
        }
        return index;
      }

      constexpr auto order() const -> int
      {
        return outer().size();
      }

      constexpr bool is_zero() const
      {
        return tag == RATIONAL && q == Rational(0);
      }

      constexpr bool is_one() const
      {
        return tag == RATIONAL && q == Rational(1);
      }

      constexpr friend bool is_equivalent(Node const* a, Node const* b)
      {
        assert(a && b);
        if (a->tag != b->tag) return false;
        if (a->index != b->index) return false;
        if (a->constant != b->constant) return false;
        switch (a->tag) { // todo: commutative
         case SUM:
         case DIFFERENCE:
         case PRODUCT:
         case RATIO:
          return (is_equivalent(a->a(), b->a()) && is_equivalent(a->b(), b->b()));
         case DOUBLE:   return (a->d == b->d);
         case RATIONAL: return (a->q == b->q);
         case TENSOR:   return (a->tensor == b->tensor);
         default: return true;
        }
      }

      constexpr auto tensors(set<Node const*>& out) const -> int
      {
        if (tag == TENSOR) {
          return (out.emplace(this));
        }

        if (tag_is_binary(tag)) {
          return a_->tensors(out) + b_->tensors(out);
        }

        return 0;
      }

      /// How many elements are in the runtime tensor for this node.
      constexpr auto tensor_size(int dim) const -> int
      {
        return ttl::pow(dim, order());
      }

      constexpr auto shape(int dim, int stack) const -> TreeShape
      {
        if (!std::is_constant_evaluated()) {
          fmt::print("{} ", stack);
        }

        switch (tag) {
         case SUM:
         case DIFFERENCE:
         case PRODUCT:
         case RATIO: {
           // reserve space for return values and call a, then b
           stack += a_->tensor_size(dim);
           TreeShape a = a_->shape(dim, stack);

           stack += b_->tensor_size(dim);
           TreeShape b = b_->shape(dim, stack);

           return TreeShape(order(), a, b);
         }

         case INDEX:
          return TreeShape(index.size(), stack, kw::n_immediates = 0);

         case DOUBLE:
         case RATIONAL:
          return TreeShape(index.size(), stack, kw::n_immediates = 1);

         case TENSOR: {
           int order = all().size();
           int n = ttl::pow(dim, order);
           return TreeShape(index.size(), stack, kw::n_scalars = n);
         }

         case PARTIAL:
          assert(false);
        }
        __builtin_unreachable();
      }

      /// Use contract to evaluate op(Scalar) for this node.
      constexpr auto scalars(int N, auto&& op) const
      {
        assert(tag == TENSOR);
        Index space = all();
        Index inner = index;

        ScalarIndex i(space.size());
        do {
          op(Scalar(N, tensor, i.select(space, inner), constant));
        } while (i.carry_sum_inc(N));
      }

      auto to_string() const -> std::string
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
        __builtin_unreachable();
      }
    };

    Tensor lhs_;
    Node* root_;

    constexpr ~TensorTree()
    {
      delete root_;
    }

    template <int M>
    constexpr TensorTree(Tensor const& lhs, ParseTree<M> const& tree, auto&& constants)
        : lhs_(lhs)
        , root_(map(tree.root(), constants))
    {
    }

    constexpr TensorTree(TensorTree const&) = delete;
    constexpr TensorTree(TensorTree&& b)
        : lhs_(b.lhs_)
        , root_(std::exchange(b.root_, nullptr))
    {
      assert(lhs_.order() == root_->order());
    }

    constexpr auto lhs() const -> Tensor const&
    {
      return lhs_;
    }

    constexpr auto root() const -> Node const*
    {
      return root_;
    }

    constexpr auto outer() const -> Index
    {
      return root_->outer();
    }

    constexpr auto order() const -> int
    {
      return outer().size();
    }

    constexpr auto tensors() const -> set<Node const*>
    {
      set<Node const*> t;
      root()->tensors(t);
      return t;
    }

    constexpr auto scalars(int N, set<Scalar>& out) const -> decltype(auto)
    {
      for (Node const* node : tensors())
      {
        assert(node->tag == TENSOR);
        node->scalars(N, [&](Scalar scalar) {
          out.emplace(std::move(scalar));
        });
      }

      ScalarIndex index(order());
      do {
        out.emplace(N, lhs_, index, false);
      } while (index.carry_sum_inc(N));

      return out;
    }

    constexpr auto shape(int dim) const -> TreeShape
    {
      // allocate space on the stack for the returned tensor
      int stack = root_->tensor_size(dim);
      auto tree = root_->shape(dim, stack);
      if (!std::is_constant_evaluated()) {
        fmt::print("{} \n", stack);
      }
      return tree;
    }

    auto to_string() const -> std::string
    {
      return std::string(lhs_.id()).append(" = ").append(root_->to_string());
    }

   private:
    constexpr static auto map(const ParseNode* node, auto const& constants) -> Node*
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

    constexpr static auto reduce(Tag tag, Node* a, Node* b) -> Node*
    {
      switch (tag) {
       case SUM:        return reduce_sum(a, b);
       case DIFFERENCE: return reduce_difference(a, b);
       case PRODUCT:    return reduce_product(a, b);
       case RATIO:      return reduce_ratio(a, b);
       default: assert(false);
      }
      __builtin_unreachable();
    }

    constexpr static auto reduce_sum(Node* a, Node* b) -> Node*
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

    constexpr static auto reduce_difference(Node* a, Node* b) -> Node*
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

    constexpr static auto reduce_product(Node* a, Node* b) -> Node*
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

    constexpr static auto reduce_ratio(Node* a, Node* b) -> Node*
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

    constexpr static auto dx(Node* node, Index const& index) -> Node*
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
      __builtin_unreachable();
    }

    constexpr static auto dx_product(Node* a, Node* b, Index const& index) -> Node*
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

    constexpr static auto dx_quotient(Node* a, Node* b, Index const& index) -> Node*
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
