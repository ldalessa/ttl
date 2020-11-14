#pragma once

#include "Tag.hpp"
#include "ttl/Index.hpp"
#include "ttl/Rational.hpp"
#include "ttl/Tensor.hpp"
#include <fmt/core.h>
#include <utility>

namespace ttl::tensor
{
struct RPNNode {
  Tag  tag;
  int    left = 0;
  constexpr static int right = 1;
  Index index = {};
  union {
    double   d = 0;
    Rational q;
    Tensor   tensor;
  };

  constexpr RPNNode() {}

  constexpr RPNNode(Tag tag, int left, const Index& i)
      : tag(tag)
      , left(left)
      , index(i)
  {
    assert(tag_is_binary(tag));
    assert(1 < left);
  }

  constexpr RPNNode(const Tensor& t, const Index& i)
      : tag(TENSOR)
      , tensor(t)
      , index(i)
  {
    assert(tensor.order() <= i.size());
  }

  constexpr RPNNode(const Index& i)
      : tag(INDEX)
      , index(i)
  {
  }

  constexpr RPNNode(const Rational& q) : tag(RATIONAL), q(q) {}
  constexpr RPNNode(std::floating_point auto d) : tag(DOUBLE),   d(d) {}
};

template <int M = 1>
struct RPNTree
{
  constexpr static std::true_type is_tree_tag = {};

  RPNNode data[M];

  constexpr RPNTree() {}
  constexpr RPNTree(Rational q)
      : data { q }
  {
  }

  constexpr RPNTree(std::signed_integral auto i)
      : data { Rational(i) }
  {
  }

  constexpr RPNTree(std::floating_point auto d)
      : data { d }
  {
  }

  constexpr RPNTree(const Index& i)
      : data { i }
  {
  }

  constexpr RPNTree(const Tensor& t, const Index& i)
      : data { {t, i} }
  {
  }

  constexpr RPNTree(const Tensor& t)
      : data { {t, Index() } }
  {
  }

  constexpr RPNTree(Tag tag, is_tree auto&& a, is_tree auto&& b)
  {
    int i = 0;
    for (auto&& node : a.data) data[i++] = node;
    for (auto&& node : b.data) data[i++] = node;
    data[i] = RPNNode(tag, b.size() + 1, tag_outer(tag, a.outer(), b.outer()));
  }


  constexpr        int   size() const { return M; }
  constexpr const RPNNode* begin() const { return data; }
  constexpr const RPNNode*   end() const { return data + size(); }

  constexpr const Index& outer() const {
    return data[M - 1].index;
  }

  constexpr const RPNNode* root() const {
    return data + M - 1;
  }

  constexpr const RPNNode* a(const RPNNode* node) const {
    return data + (node - data) - node->left;
  }

  constexpr const RPNNode* b(const RPNNode* node) const {
    return data + (node - data) - node->right;
  }
};

template <int A, int B>
RPNTree(Tag, RPNTree<A>, RPNTree<B>) -> RPNTree<A + B + 1>;
}

template <>
struct fmt::formatter<ttl::tensor::RPNNode>
{
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::tensor::RPNNode& node, FormatContext& ctx)
  {
    using namespace ttl::tensor;
    switch (node.tag) {
     case SUM:
     case DIFFERENCE:
     case PRODUCT:
     case RATIO:
     case PARTIAL:  return format_to(ctx.out(), "{}", node.tag);
     case INDEX:    return format_to(ctx.out(), "{}", node.index);
     case RATIONAL: return format_to(ctx.out(), "{}", node.q);
     case DOUBLE:   return format_to(ctx.out(), "{}", node.d);
     case TENSOR:
      if (node.index.size()) {
        return format_to(ctx.out(), "{}({})", node.tensor, node.index);
      }
      else {
        return format_to(ctx.out(), "{}", node.tensor);
      }
     default:
      __builtin_unreachable();
    }
  }
};

template <int N>
struct fmt::formatter<ttl::tensor::RPNTree<N>> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::tensor::RPNTree<N>& tree, FormatContext& ctx)
  {
    using namespace ttl::tensor;
    auto op = [&](const RPNNode* node, auto&& self) -> std::string {
      switch (node->tag) {
       case SUM:
       case DIFFERENCE:
       case PRODUCT:
       case RATIO: {
        std::string b = self(tree.b(node), self);
        std::string a = self(tree.a(node), self);
        return fmt::format("({} {} {})", a, *node, b);
       }

       case PARTIAL: {
        std::string b = self(tree.b(node), self);
        std::string a = self(tree.a(node), self);
        return fmt::format("D({},{}", a, b);
       }

       case INDEX:
       case TENSOR:
       case RATIONAL:
       case DOUBLE:
        return fmt::format("{}", *node);

       default: assert(false);
      }
    };
    return format_to(ctx.out(), "{}", op(tree.root(), op));
  }
};
