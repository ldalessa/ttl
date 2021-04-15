#pragma once

#include "Index.hpp"
#include "Rational.hpp"
#include "Tag.hpp"
#include "Tensor.hpp"
#include <utility>

namespace ttl
{
  struct ParseNode
  {
    Tag     tag = {};
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

    constexpr ParseNode(const Rational& q)
        : tag(RATIONAL)
        , q(q)
    {
    }

    constexpr ParseNode(std::floating_point auto d)
        : tag(DOUBLE)
        , d(d)
    {
    }

    constexpr auto outer() const -> Index const&
    {
      return index;
    }

    constexpr auto a() const -> ParseNode const*
    {
      return this - left;
    }

    constexpr auto b() const -> ParseNode const*
    {
      return this - right;
    }

    auto to_string() const -> std::string
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
    using is_tree_tag = void;

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

    constexpr auto operator()(std::same_as<Index> auto const&... is) const
      -> ParseTree
    {
      Index  search(outer());
      Index replace(is...);
      assert(search.size() == replace.size());

      ParseTree copy(*this);
      for (ParseNode& node : copy.data) {
        node.index.search_and_replace(search, replace);
      }
      return copy;
    }

    constexpr auto size() const -> int
    {
      return M;
    }

    constexpr auto begin() const
    {
      return std::begin(data);
    }

    constexpr auto end() const
    {
      return begin() + size();
    }

    constexpr auto outer() const -> Index
    {
      return exclusive(data[M - 1].index);
    }

    constexpr auto order() const -> int
    {
      return outer().size();
    }

    constexpr auto root() const -> ParseNode const*
    {
      return data + M - 1;
    }

    auto to_string() const -> std::string
    {
      return data[M - 1].to_string();
    }
  };

  template <int A, int B>
  ParseTree(Tag, ParseTree<A>, ParseTree<B>) -> ParseTree<A + B + 1>;
}
