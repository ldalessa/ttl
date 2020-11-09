#pragma once

#include "Index.hpp"
#include "Rational.hpp"
#include "Tags.hpp"
#include "Tensor.hpp"
#include "utils.hpp"

namespace ttl {
template <int N = 1>
struct Tree final
{
  constexpr static std::true_type is_tree_tag = {};
  Node nodes[N];

  constexpr Tree() = default;

  constexpr Tree(Index index, Tag tag = INDEX) {
    nodes[0] = Node(index, tag);
  }

  constexpr Tree(const Tensor* t) {
    nodes[0] = Node(t);
  }

  constexpr Tree(Rational q) {
    nodes[0] = Node(q);
  }

  constexpr Tree(std::signed_integral auto i) {
    nodes[0] = Node(Rational(i));
  }

  constexpr Tree(std::floating_point auto d) {
    nodes[0] = Node(d);
  }

  template <int A, int B>
  constexpr Tree(Tag tag, const Tree<A>& a, const Tree<B>& b) {
    assert(A + B + 1 == N);
    assert(binary(tag));
    std::copy_n(a.nodes, A, nodes);
    std::copy_n(b.nodes, B, nodes + A);
    nodes[N - 1] = Node(outer(tag, outer(a), outer(b)), tag);
  }

  constexpr const Node& operator[](int i) const { return nodes[i]; }
  constexpr       Node& operator[](int i)       { return nodes[i]; }

  constexpr const Node* begin() const { return nodes; }
  constexpr const Node*   end() const { return nodes + N; }
  constexpr       Node* begin()       { return nodes; }
  constexpr       Node*   end()       { return nodes + N; }

  constexpr Tree operator()(std::same_as<Index> auto... is) const {
    Tree copy = *this;
    Index   search = outer(copy);
    Index  replace = {is...};
    assert(search.size() == replace.size());
    for (Node& node : copy) {
      if (Index* index = node.index()) {
        index->search_and_replace(search, replace);
      }
    }

    return copy;
  }

  constexpr friend int size(const Tree&) {
    return N;
  }

  constexpr friend Index outer(const Tree& tree) {
    if (const Index* index = tree.nodes[N-1].index()) {
      return *index;
    }
    return {};
  }

  constexpr Tag tag(int i) const {
    return nodes[i].tag;
  }
};

template <int A, int B>
Tree(Tag, Tree<A>, Tree<B>) -> Tree<A + B + 1>;

template <typename T>
concept is_expression =
 is_tree<T> ||
 std::same_as<T, Tensor> ||
 std::same_as<T, Rational> ||
 std::signed_integral<T> ||
 std::floating_point<T>;

constexpr auto bind(const Tensor& t) {
  return Tree(std::addressof(t));
}

template <int N>
constexpr Tree<N>&& bind(Tree<N>&& a) {
  return a;
}

template <int N>
constexpr auto bind(const Tree<N>& a) {
  return a;
}

constexpr auto bind(is_expression auto const& a) {
  return Tree(a);
}

constexpr auto operator+(is_expression auto const& a) {
  return bind(a);
}

constexpr auto operator+(is_expression auto const& a, is_expression auto const& b) {
  return Tree(SUM, bind(a), bind(b));
}

constexpr auto operator*(is_expression auto const& a, is_expression auto const& b) {
  return Tree(PRODUCT, bind(a), bind(b));
}

constexpr auto operator-(is_expression auto const& a, is_expression auto const& b) {
  return Tree(DIFFERENCE, bind(a), bind(b));
}

constexpr auto operator-(is_expression auto const& a) {
  return Rational(-1) * bind(a);
}

constexpr auto operator/(is_expression auto const& a, is_expression auto const& b) {
  return Tree(RATIO, bind(a), bind(b));
}

constexpr auto D(is_expression auto const& a, std::same_as<Index> auto... is) {
  return Tree(PARTIAL, bind(a), Tree((is + ...)));
}

constexpr Tree<1> delta(const Index& a, const Index& b) {
  assert(a.size() == 1);
  assert(b.size() == 1);
  assert(a != b);
  return Tree(a + b, DELTA);
}

constexpr auto symmetrize(is_expression auto const& a) {
  Tree t = bind(a);
  return Rational(1,2) * (t + t(reverse(outer(t))));
}

constexpr auto Tensor::operator()(std::same_as<Index> auto... is) const {
  return Tree(BIND, Tree(this), Tree((is + ...)));
}
}


template <int N>
struct fmt::formatter<ttl::Tree<N>> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::Tree<N>& tree, FormatContext& ctx)
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
        stack.push(fmt::to_string(node));
        break;
       default:
        __builtin_unreachable();
      }
    }
    return format_to(ctx.out(), "{}", stack.pop());
  }
};
