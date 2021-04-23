#pragma once

#include "ttl/Index.hpp"
#include "ttl/Rational.hpp"
#include "ttl/ScalarIndex.hpp"
#include "ttl/Tag.hpp"
#include "ttl/Tensor.hpp"
#include "ttl/concepts.hpp"
#include "ce/dvector.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
#include <span>
#include <string>

#define ttl_assert(op, ...) assert((op))

namespace ttl
{

  template <int M>
  struct ParseTree
  {
    using is_parse_tree_tag = void;

    Tag                 tags[M];
    int                left_[M];
    Rational              qs[M];
    double                ds[M];
    Tensor const*    tensors[M];
    Index       tensor_index[M];
    ScalarIndex scalar_index[M];

    int    size_= M;
    int   depth_ = 1;
    Index outer_ = {};

    template <int A, int B>
    constexpr ParseTree(ParseTree<A> const& a, ParseTree<B> const& b, Tag tag)
        : depth_ { std::max(a.depth_, b.depth_) + 1 }
        , outer_ { tag.outer(a.outer_, b.outer_) }
    {
      assert(tag.is_binary());

      auto join = [](auto* a, auto* b, auto* c) {
        return std::copy(b, b + B, std::copy(a, a + A, c));
      };

      *join(a.tags, b.tags, tags) = tag;
      *join(a.left_, b.left_, left_) = M - A;
      *join(a.qs, b.qs, qs) = {};
      *join(a.ds, b.ds, ds) = {};
      *join(a.tensors, b.tensors, tensors) = {};
      *join(a.tensor_index, b.tensor_index, tensor_index) = {};
      *join(a.scalar_index, b.scalar_index, scalar_index) = {};

      if (tag == SUM or tag == DIFFERENCE) {
        ttl_assert(permutation(a.outer_, b.outer_), "index mismatch for + or - operation");
      }
    }

    constexpr ParseTree(ParseTree<M - 1> const& a, Tag tag)
        : depth_ { a.depth_ + 1 }
        , outer_ { tag.outer(a.outer_, {}) }
    {
      assert(tag.is_unary());

      using std::begin;
      using std::end;
      using std::copy;

      *copy(begin(a.tags), end(a.tags), tags) = tag;
      *copy(begin(a.left_), end(a.left_), left_) = 0;
      *copy(begin(a.qs), end(a.qs), qs) = {};
      *copy(begin(a.ds), end(a.ds), ds) = {};
      *copy(begin(a.tensors), end(a.tensors), tensors) = {};
      *copy(begin(a.tensor_index), end(a.tensor_index), tensor_index) = {};
      *copy(begin(a.scalar_index), end(a.scalar_index), scalar_index) = {};
    }

    constexpr ParseTree(ParseTree<M - 1> const& a, Tag tag, Index const& i)
        : depth_ { a.depth_ + 1 }
        , outer_ { tag.outer(a.outer_, i) }
    {
      assert(tag == PARTIAL || tag == BIND);

      using std::begin;
      using std::end;
      using std::copy;

      *copy(begin(a.tags), end(a.tags), tags) = tag;
      *copy(begin(a.left_), end(a.left_), left_) = 0;
      *copy(begin(a.qs), end(a.qs), qs) = {};
      *copy(begin(a.ds), end(a.ds), ds) = {};
      *copy(begin(a.tensors), end(a.tensors), tensors) = {};
      *copy(begin(a.tensor_index), end(a.tensor_index), tensor_index) = { i };
      *copy(begin(a.scalar_index), end(a.scalar_index), scalar_index) = {};
    }

    constexpr ParseTree(Tensor const* a, Index const& i)
        : tags    { TENSOR }
        , left_   { 0 }
        , qs      {}
        , ds      {}
        , tensors { a }
        , tensor_index { i }
        , scalar_index {}
        , outer_ { tags[0].outer(i, {}) }
    {
      assert(a);
    }

    constexpr ParseTree(Tensor const* a, ScalarIndex const& i)
        : tags    { SCALAR }
        , left_   { 0 }
        , qs      {}
        , ds      {}
        , tensors { a }
        , tensor_index {}
        , scalar_index { i }
    {
      assert(a);
    }

    constexpr ParseTree(Tag tag, Index const& i)
        : tags    { tag }
        , left_   { 0 }
        , qs      {}
        , ds      {}
        , tensors {}
        , tensor_index { i }
        , scalar_index {}
        , outer_ { tag.outer(i, {}) }
    {
      assert(tag == DELTA || tag == EPSILON);
    }

    constexpr ParseTree(Rational const& q)
        : tags    { RATIONAL }
        , left_   { 0 }
        , qs      { q }
        , ds      {}
        , tensors {}
        , tensor_index {}
        , scalar_index {}
    {
    }

    constexpr ParseTree(double d)
        : tags    { DOUBLE }
        , left_   { 0 }
        , qs      {}
        , ds      { d }
        , tensors {}
        , tensor_index {}
        , scalar_index {}
    {
    }

    constexpr auto operator()(is_index auto... is) const
      -> ParseTree<M+1>
    {
      return ParseTree<M+1>(*this, BIND, (is + ... + Index{}));
    }

    /// Return the tag for the root of the
    constexpr auto tag() const -> Tag
    {
      return tags[M - 1];
    }

    /// Return the tensor index stored for any node in the tree.
    constexpr auto index(int i) const -> Index
    {
      return tensor_index[i];
    }

    constexpr auto left(int i = M - 1) const -> int
    {
      assert(0 < left_[i]);
      return i - left_[i];
    }

    constexpr auto right(int i = M - 1) const -> int
    {
      return i - 1;
    }

    /// Return the cached outer index for the tree root.
    constexpr auto outer() const -> Index
    {
      return outer_;
    }

    /// Compute the outer index for any node in the tree.
    constexpr auto outer(int k) const -> Index
    {
      ce::dvector<Index> stack; stack.reserve(depth_);
      for (int i = 0; i < k + 1; ++i)
      {
        Tag tag = tags[i];
        if (tag.is_binary()) {
          Index b = stack.pop_back();
          Index a = stack.pop_back();
          stack.push_back(tag.outer(a, b));
        }
        else if (tag.is_unary()) {
          Index a = stack.pop_back();
          stack.push_back(tag.outer(a, tensor_index[i]));
        }
        else {
          stack.push_back(tag.outer(tensor_index[i]));
        }
      }
      return stack.pop_back();
    }

    constexpr friend auto outer(ParseTree const& tree) -> Index
    {
      return tree.outer();
    }

    auto to_string() const -> std::string
    {
      ce::dvector<std::string> stack; stack.reserve(depth_);
      for (int i = 0; i < size_; ++i) {
        switch (tags[i])
        {
          // Binary infix are the same.
         case SUM:
         case DIFFERENCE:
         case PRODUCT:
         case RATIO: {
           std::string b = stack.pop_back();
           std::string a = stack.pop_back();
           std::string c = fmt::format("({} {} {})", std::move(a), tags[i], std::move(b));
           stack.emplace_back(std::move(c));
         } continue;

         case BIND:
         case PARTIAL: {
           std::string a = stack.pop_back();
           std::string b = fmt::format("{}({},{})", tags[i], std::move(a), tensor_index[i]);
           stack.emplace_back(std::move(b));
         } continue;

         case POW: {
           std::string b = stack.pop_back();
           std::string a = stack.pop_back();
           std::string c = fmt::format("({}){}{}", std::move(a), tags[i], std::move(b));
           stack.emplace_back(std::move(c));
         } continue;

          // All the functions are the same.
         case SQRT:
         case EXP:
         case DELTA:
         case EPSILON: {
           stack.emplace_back(fmt::format("{}({})", tags[i], tensor_index[i]));
         } continue;

         case NEGATE: {
           std::string a = stack.pop_back();
           std::string b = fmt::format("{}{}", tags[i], std::move(a));
           stack.emplace_back(std::move(b));
         } continue;

         case RATIONAL: {
           stack.emplace_back(fmt::format("{}", qs[i]));
         } continue;

         case DOUBLE: {
           stack.emplace_back(fmt::format("{}", ds[i]));
         } continue;

         case TENSOR: {
           if (tensor_index[i].size()) {
             stack.emplace_back(fmt::format("{}({})", *tensors[i], tensor_index[i]));
           }
           else {
             stack.emplace_back(fmt::format("{}", *tensors[i]));
           }
         } continue;

         case SCALAR: {
           if (scalar_index[i].size()) {
             stack.emplace_back(fmt::format("{}", *tensors[i], scalar_index[i]));
           }
           else {
             stack.emplace_back(fmt::format("{}({})", *tensors[i]));
           }
         } continue;
        }
        __builtin_unreachable();
      }
      assert(stack.size() == 1);
      return stack.pop_back();
    }

    friend auto to_string(ParseTree<M> const& tree) -> std::string
    {
      return tree.to_string();
    }

    auto to_dot(auto&& out) const -> auto&
    {
      for (int i = 0; i < size_; ++i)
      {
        Index outer = this->outer(i);
        switch (Tag tag = tags[i])
        {
          // Binary infix are the same.
         case SUM:
         case DIFFERENCE:
         case PRODUCT:
         case RATIO:
         case POW: {
           if (outer.size()) {
             fmt::format_to(out, "\tnode{}[label=\"{} ↑{}\"]\n", i, tag, outer);
           }
           else {
             fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, tag, outer);
           }
           fmt::format_to(out, "\tnode{} -- node{}\n", i, left(i));
           fmt::format_to(out, "\tnode{} -- node{}\n", i, right(i));
         } continue;

         case BIND:
         case PARTIAL: {
           if (outer.size()) {
             fmt::format_to(out, "\tnode{}[label=\"{}({},{}) ↑{}\"]\n", i, tag, this->outer(i-1), index(i), outer);
           }
           else {
             fmt::format_to(out, "\tnode{}[label=\"{}({},{})\"]\n", i, tag, this->outer(i-1), index(i));
           }
           fmt::format_to(out, "\tnode{} -- node{}\n", i, right(i));
         } continue;

          // All the functions are the same.
         case SQRT:
         case EXP:
         case NEGATE: {
           fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, tag);
           fmt::format_to(out, "\tnode{} -- node{}\n", i, right(i));
         } continue;

         case RATIONAL: {
             fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, qs[i]);
         } continue;

         case DOUBLE: {
             fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, ds[i]);
         } continue;

         case TENSOR: {
           if (tensor_index[i].size()) {
             fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, *tensors[i], tensor_index[i]);
           }
           else {
             fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, *tensors[i]);
           }
         } continue;

         case SCALAR: {
           if (scalar_index[i].size()) {
             fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, *tensors[i], scalar_index[i]);
           }
           else {
             fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, *tensors[i]);
           }
         } continue;

         case DELTA:
         case EPSILON: {
           fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, tag, tensor_index[i]);
         } continue;
        }
        __builtin_unreachable();
      }
      return out;
    }
  };

  template <int A>
  ParseTree(ParseTree<A>, Tag, auto...) -> ParseTree<A + 1>;

  template <int A, int B>
  ParseTree(ParseTree<A>, ParseTree<B>, Tag) -> ParseTree<A + B + 1>;

  constexpr auto Tensor::bind_tensor(is_index auto... is) const
  {
    return ParseTree<1>(this, (is + ... + Index{}));
  }

    constexpr auto Tensor::bind_scalar(std::signed_integral auto... is) const
  {
    return ParseTree<1>(this, ScalarIndex(std::in_place, is...));
  }

  template <class T>
  concept is_parse_expression =
  is_parse_tree<T> ||
  std::same_as<T, Rational> ||
  std::same_as<T, Tensor> ||
  std::signed_integral<T> ||
  std::floating_point<T>;

  constexpr auto bind(is_parse_tree auto const& tree)
    -> is_parse_tree auto const&
  {
    return tree;
  }

  constexpr auto bind(Rational const& q)
    -> ParseTree<1>
  {
    return ParseTree<1>(q);
  }

  constexpr auto bind(Tensor const& t)
    -> ParseTree<1>
  {
    assert(t.order() == 0);
    return ParseTree<1>(&t, Index{});
  }

  constexpr auto bind(std::signed_integral auto i)
    -> ParseTree<1>
  {
    return bind(Rational(i));
  }

  constexpr auto bind(std::floating_point auto d)
    -> ParseTree<1>
  {
    return ParseTree<1>(d);
  }

  constexpr auto operator+(is_parse_expression auto const& a)
    -> is_parse_expression auto const&
  {
    return a;
  }

  constexpr auto operator+(is_parse_expression auto const& a,
                           is_parse_expression auto const& b)
  {
    return ParseTree(bind(a), bind(b), SUM);
  }

  constexpr auto operator-(is_parse_expression auto const& a)
  {
    return ParseTree(bind(a), NEGATE);
  }

  constexpr auto operator-(is_parse_expression auto const& a,
                           is_parse_expression auto const& b)
  {
    return ParseTree(bind(a), bind(b), DIFFERENCE);
  }

  constexpr auto operator*(is_parse_expression auto const& a,
                           is_parse_expression auto const& b)
  {
    return ParseTree(bind(a), bind(b), PRODUCT);
  }

  constexpr auto operator/(is_parse_expression auto const& a,
                           is_parse_expression auto const& b)
  {
    return ParseTree(bind(a), bind(b), RATIO);
  }

  constexpr auto D(is_parse_expression auto const& a,
                   Index i, is_index auto... is)
  {
    return ParseTree(bind(a), PARTIAL, (i + ... + is));
  }

  constexpr auto δ(Index i, Index j)
  {
    assert(i.size() == 1);
    assert(j.size() == 1);
    return ParseTree<1>(DELTA, i + j);
  }

  constexpr auto symmetrize(is_parse_expression auto const& a)
  {
    auto&& tree = bind(a);
    return Rational(1,2) * (tree + tree(reverse(outer(tree))));
  }

  // constexpr auto exp(is_parse_expression auto const& a)
  // {
  //   return ParseTree(to_parse_tree(a), EXP);
  // }

  // constexpr auto pow(is_parse_expression auto const& a, std::integral auto b)
  // {
  //   return ParseTree(to_parse_tree(a), to_parse_tree(b), POW);
  // }

  // constexpr auto sqrt(is_parse_expression auto const& a)
  // {
  //   return ParseTree(to_parse_tree(a), SQRT);
  // }
}

