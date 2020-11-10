#pragma once

#include "Index.hpp"
#include "Node.hpp"
#include "Rational.hpp"
#include "Tensor.hpp"
#include "utils.hpp"
#include <concepts>
#include <string>
#include <fmt/core.h>

namespace ttl {
struct TensorTreeNode : ttl::Node {
  int left = 0;
  static constexpr int right = 1;
  using ttl::Node::Node;
};

template <int N = 1>
struct TensorTree final
{
  using Node = TensorTreeNode;

  constexpr static std::true_type is_tree_tag = {};
  Node nodes[N];

  constexpr TensorTree() = default;

  constexpr TensorTree(Index index, Tag tag = INDEX) {
    nodes[0] = Node(index, tag);
  }

  constexpr TensorTree(const Tensor* t) {
    nodes[0] = Node(t);
  }

  constexpr TensorTree(Rational q) {
    nodes[0] = Node(q);
  }

  constexpr TensorTree(std::signed_integral auto i) {
    nodes[0] = Node(Rational(i));
  }

  constexpr TensorTree(std::floating_point auto d) {
    nodes[0] = Node(d);
  }

  template <int A, int B>
  constexpr TensorTree(Tag tag, const TensorTree<A>& a, const TensorTree<B>& b) {
    assert(A + B + 1 == N);
    assert(binary(tag));
    int i = 0;
    for (auto&& a : a) nodes[i++] = a;
    for (auto&& b : b) nodes[i++] = b;
    nodes[i] = Node(outer(tag, outer(a), outer(b)), tag);
    nodes[i].left = B + 1;
  }

  constexpr const Node& operator[](int i) const { return nodes[i]; }
  constexpr       Node& operator[](int i)       { return nodes[i]; }

  constexpr const Node* begin() const { return nodes; }
  constexpr const Node*   end() const { return nodes + N; }
  constexpr       Node* begin()       { return nodes; }
  constexpr       Node*   end()       { return nodes + N; }

  constexpr TensorTree operator()(std::same_as<Index> auto... is) const {
    TensorTree copy = *this;
    Index    search = outer(copy);
    Index   replace = {is...};
    assert(search.size() == replace.size());
    for (Node& node : copy) {
      if (Index* index = node.index()) {
        index->search_and_replace(search, replace);
      }
    }

    return copy;
  }

  constexpr friend int size(const TensorTree&) {
    return N;
  }

  constexpr friend Index outer(const TensorTree& tree) {
    if (const Index* index = tree.nodes[N-1].index()) {
      return *index;
    }
    return {};
  }

  constexpr Tag tag(int i) const {
    return nodes[i].tag;
  }

  constexpr const Node& root() const {
    return nodes[N - 1];
  }

  constexpr const Node& a(const Node& node) const {
    assert(node.binary());
    assert(node.left > 0);
    return nodes[&node - nodes - node.left];
  }

  constexpr const Node& b(const Node& node) const {
    assert(node.binary());
    assert(node.right == 1);
    return nodes[&node - nodes - node.right];
  }
};

template <int A, int B>
TensorTree(Tag, TensorTree<A>, TensorTree<B>) -> TensorTree<A + B + 1>;

template <typename T>
concept is_expression =
 is_tree<T> ||
 std::same_as<T, Tensor> ||
 std::same_as<T, Rational> ||
 std::signed_integral<T> ||
 std::floating_point<T>;

constexpr auto bind(const Tensor& t) {
  return TensorTree(std::addressof(t));
}

template <int N>
constexpr TensorTree<N>&& bind(TensorTree<N>&& a) {
  return a;
}

template <int N>
constexpr auto bind(const TensorTree<N>& a) {
  return a;
}

constexpr auto bind(is_expression auto const& a) {
  return TensorTree(a);
}

constexpr auto operator+(is_expression auto const& a) {
  return bind(a);
}

constexpr auto operator+(is_expression auto const& a, is_expression auto const& b) {
  return TensorTree(SUM, bind(a), bind(b));
}

constexpr auto operator*(is_expression auto const& a, is_expression auto const& b) {
  return TensorTree(PRODUCT, bind(a), bind(b));
}

constexpr auto operator-(is_expression auto const& a, is_expression auto const& b) {
  return TensorTree(DIFFERENCE, bind(a), bind(b));
}

constexpr auto operator-(is_expression auto const& a) {
  return Rational(-1) * bind(a);
}

constexpr auto operator/(is_expression auto const& a, is_expression auto const& b) {
  return TensorTree(RATIO, bind(a), bind(b));
}

constexpr auto D(is_expression auto const& a, std::same_as<Index> auto... is) {
  return TensorTree(PARTIAL, bind(a), TensorTree((is + ...)));
}

constexpr TensorTree<1> delta(const Index& a, const Index& b) {
  assert(a.size() == 1);
  assert(b.size() == 1);
  assert(a != b);
  return TensorTree(a + b, DELTA);
}

constexpr auto symmetrize(is_expression auto const& a) {
  TensorTree t = bind(a);
  return Rational(1,2) * (t + t(reverse(outer(t))));
}

constexpr auto Tensor::operator()(std::same_as<Index> auto... is) const {
  return TensorTree(BIND, TensorTree(this), TensorTree((is + ...)));
}
}

template <int N>
struct fmt::formatter<ttl::TensorTree<N>> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::TensorTree<N>& tree, FormatContext& ctx)
  {
    ttl::utils::stack<std::string> stack;
    for (auto&& node : tree) {
      switch (node.tag) {
       case ttl::SUM:
       case ttl::DIFFERENCE:
       case ttl::PRODUCT:
       case ttl::RATIO:
        stack.push(fmt::format("({1} {2} {0})", stack.pop(), stack.pop(), node));
        break;
       case ttl::BIND:
        stack.push(fmt::format("{1}({0})", stack.pop(), stack.pop()));
        break;
       case ttl::PARTIAL:
        stack.push(fmt::format("D({1},{0})", stack.pop(), stack.pop()));
        break;
       case ttl::DELTA:
        stack.push(fmt::format("delta({})", node));
        break;
       case ttl::INDEX:

       case ttl::TENSOR:
       case ttl::RATIONAL:
       case ttl::DOUBLE:
        stack.push(fmt::format("{}", node));
        break;
       default:
        __builtin_unreachable();
      }
    }
    return format_to(ctx.out(), "{}", stack.pop());
  }
};
