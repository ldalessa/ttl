#pragma once

#include "Index.hpp"
#include "Rational.hpp"
#include "Tag.hpp"
#include "Tensor.hpp"
#include <utility>

namespace ttl
{
struct ParseNode {
  Tag     tag;
  int    left = 0;
  constexpr static int right = 1;
  Index index = {};
  union {
    double   d = 0;
    Rational q;
    Tensor   tensor;
  };

  constexpr ParseNode() {}

  constexpr ParseNode(Tag tag, int left, const Index& i)
      : tag(tag)
      , left(left)
      , index(i)
  {
    assert(tag_is_binary(tag));
    assert(1 < left);
  }

  constexpr ParseNode(const Tensor& t, const Index& i)
      : tag(TENSOR)
      , tensor(t)
      , index(i)
  {
    assert(tensor.order() <= i.size());
  }

  constexpr ParseNode(const Index& i)
      : tag(INDEX)
      , index(i)
  {
  }

  constexpr ParseNode(const Rational& q) : tag(RATIONAL), q(q) {}
  constexpr ParseNode(std::floating_point auto d) : tag(DOUBLE),   d(d) {}

  constexpr const Index& outer() const {
    return index;
  }

  constexpr const ParseNode* a() const {
    return this - left;
  }

  constexpr const ParseNode* b() const {
    return this - right;
  }

  std::string to_string() const
  {
    switch (tag)
    {
     case SUM:
     case DIFFERENCE:
     case PRODUCT:
     case RATIO:
      return fmt::format("({} {} {})", a()->to_string(), tag, b()->to_string());

     case PARTIAL:
      return fmt::format("D({},{})", a()->to_string(), b()->to_string());

     case INDEX:
      return fmt::format("{}", index);

     case TENSOR:
      if (index.size()) {
        return fmt::format("{}({})", tensor, index);
      }
      else {
        return fmt::format("{}", tensor);
      }

     case RATIONAL:
      return fmt::format("{}", q);

     case DOUBLE:
      return fmt::format("{}", d);

     default: assert(false);
    }
  }
};

template <int M = 1>
struct ParseTree
{
  constexpr static std::true_type is_tree_tag = {};

  ParseNode data[M];

  constexpr ParseTree() {}
  constexpr ParseTree(Rational q)
      : data { q }
  {
  }

  constexpr ParseTree(std::signed_integral auto i)
      : data { Rational(i) }
  {
  }

  constexpr ParseTree(std::floating_point auto d)
      : data { d }
  {
  }

  constexpr ParseTree(const Index& i)
      : data { i }
  {
  }

  constexpr ParseTree(const Tensor& t, const Index& i)
      : data { {t, i} }
  {
  }

  constexpr ParseTree(const Tensor& t)
      : data { {t, Index() } }
  {
  }

  constexpr ParseTree(Tag tag, is_tree auto&& a, is_tree auto&& b)
  {
    int i = 0;
    for (auto&& node : a.data) data[i++] = node;
    for (auto&& node : b.data) data[i++] = node;
    data[i] = ParseNode(tag, b.size() + 1, tag_outer(tag, a.outer(), b.outer()));
  }

  constexpr ParseTree operator()(std::same_as<Index> auto const&... is) const {
    Index  search(outer());
    Index replace(is...);
    assert(search.size() == replace.size());

    ParseTree copy(*this);
    for (ParseNode& node : copy.data) {
      node.index.search_and_replace(search, replace);
    }
    return copy;
  }

  constexpr             int   size() const { return M; }
  constexpr const ParseNode* begin() const { return data; }
  constexpr const ParseNode*   end() const { return data + size(); }

  constexpr Index outer() const {
    return exclusive(data[M - 1].index);
  }

  constexpr int order() const {
    return outer().size();
  }

  constexpr const ParseNode* root() const {
    return data + M - 1;
  }

  std::string to_string() const {
    return data[M - 1].to_string();
  }
};

template <int A, int B>
ParseTree(Tag, ParseTree<A>, ParseTree<B>) -> ParseTree<A + B + 1>;
}
