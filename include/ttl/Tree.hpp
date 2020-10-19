#pragma once

#include "Index.hpp"
#include "Tensor.hpp"

#include <ce/cvector.hpp>
#include <fmt/format.h>
#include <cassert>
#include <concepts>
#include <string_view>

namespace ttl {
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

enum Tag {
  SUM,
  DIFFERENCE,
  PRODUCT,
  INVERSE,
  BIND,
  PARTIAL,
  INDEX,
  TENSOR,
  RATIONAL,
  DOUBLE
};

struct Node {
  Tag tag_;

  union {
    Index   index_;
    Tensor tensor_;
    Rational    q_;
    double      d_;
  };

  constexpr Node(Index index, Tag tag = INDEX) : tag_(tag), index_(index) {}
  constexpr Node(const Tensor& tensor) : tag_(TENSOR), tensor_(tensor) {}
  constexpr Node(Rational q) : tag_(RATIONAL), q_(q) {}
  constexpr Node(double d) : tag_(DOUBLE), d_(d) {}

  constexpr bool is(Tag tag) const {
    return tag == tag_;
  }

  constexpr bool is_binary() const {
    return tag_ < INDEX;
  }

  constexpr const Index* index() const {
    return (tag_ < TENSOR) ? &index_ : nullptr;
  }

  constexpr Index* index() {
    return (tag_ < TENSOR) ? &index_ : nullptr;
  }

  constexpr Tensor* tensor() {
    return (tag_ == TENSOR) ? &tensor_ : nullptr;
  }

  constexpr const Tensor* tensor() const {
    return (tag_ == TENSOR) ? &tensor_ : nullptr;
  }
};

template <int M = 1>
struct Tree
{
  constexpr friend std::true_type is_tree_v(Tree) { return {}; }

  ce::cvector<int, M> left;
  ce::cvector<Node, M> nodes;

  constexpr Tree() noexcept = default;
  constexpr Tree(const Tree&) noexcept = default;
  constexpr Tree(Tree&&) noexcept = default;
  constexpr Tree& operator=(const Tree&) noexcept = default;
  constexpr Tree& operator=(Tree&&) noexcept = default;

  constexpr Tree(Tensor tensor) noexcept
      :  left(std::in_place, -1)
      , nodes(std::in_place, tensor)
  {
  }

  constexpr Tree(Index index) noexcept
      :  left(std::in_place, -1)
      , nodes(std::in_place, index)
  {
  }

  constexpr Tree(Rational q) noexcept
      :  left(std::in_place, -1)
      , nodes(std::in_place, q)
  {
  }

  constexpr Tree(double d) noexcept
      :  left(std::in_place, -1)
      , nodes(std::in_place, d)
  {
  }

  template <int A, int B>
  constexpr Tree(Tag tag, Tree<A> a, Tree<B> b) noexcept
  {
    for (int i = 0; i < a.size(); ++i) {
      left.push_back(a.left[i]);
      nodes.push_back(a.nodes[i]);
    }
    for (int i = 0; i < b.size(); ++i) {
      left.push_back(b.left[i] + a.size());
      nodes.push_back(b.nodes[i]);
    }
    left.push_back(a.size() - 1);

    Index l = a.outer();
    Index r = b.outer();
    switch (tag) {
     case SUM:
     case DIFFERENCE:
      assert(permutation(l, r));
      nodes.emplace_back(l, tag);
      break;
     case PRODUCT:
     case INVERSE:
      nodes.emplace_back(l ^ r, tag);
      break;
     case BIND:
     case PARTIAL:
      nodes.emplace_back(exclusive(l + r), tag);
      break;
     default: __builtin_unreachable();
    }
  }

  constexpr static int capacity() {
    return M;
  }

  constexpr int size() const {
    return left.size();
  }

  constexpr void rewrite(Index replace)
  {
    // Rewrite the outer index of a tree, this is a bit greedy in that there may
    // be some indices completely contracted within the tree that use the same
    // indices as the outer() index, and those will be rewritten too, but it
    // won't have an impact on the correctness of the tree.
    Index search = outer();
    assert(replace.size() == search.size());
    for (auto &&node : nodes) {
      if (Index *idx = node.index()) {
        idx->search_and_replace(search, replace);
      }
    }
  }

  constexpr Tree operator()(Index i, std::same_as<Index> auto... is) const {
    Tree copy(*this);
    copy.rewrite((i + ... + is));
    return copy;
  }

  constexpr Index outer() const {
    if (const Index *i = nodes.back().index()) {
      return *i;
    }
    return {};
  }

  // Apply the ops in a postfix traversal. Leaves are applied as op(node) while
  // internal nodes are applied recursively as op(node, op(left), op(right))
  // ish.
  constexpr auto postfix(auto&&... ops) const {
    overloaded op = { ops... };
    using T = decltype(op(nodes[0]));
    auto handler = [&](int i, auto&& Y) -> T {
      const Node& node = nodes[i];
      if (node.is_binary()) {
        return op(node, Y(left[i], Y), Y(i - 1, Y));
      }
      return op(node);
    };
    return handler(size() - 1, handler);
  }
};

template <int A, int B>
Tree(Tag, Tree<A>, Tree<B>) -> Tree<A + B + 1>;

template <typename T>
concept is_tree = requires (T t) {
  { is_tree_v(t) };
};

template <typename T>
concept is_expression =
 is_tree<T> ||
 std::same_as<T, Tensor> ||
 std::same_as<T, Index> ||
 std::same_as<T, Rational> ||
 std::signed_integral<T> ||
 std::same_as<T, double>;

constexpr auto bind(std::signed_integral auto i) {
  return Tree(Rational(i));
}

constexpr auto bind(const Tensor& t) {
  assert(t.order() == 0);
  return Tree(BIND, Tree(t), Tree(Index()));
}

constexpr auto bind(is_tree auto tree) {
  return tree;
}

constexpr auto bind(is_expression auto e) {
  return Tree(e);
}

constexpr auto operator+(is_expression auto a) {
  return bind(a);
}

constexpr auto operator+(is_expression auto a, is_expression auto b) {
  return Tree(SUM, bind(a), bind(b));
}

constexpr auto operator*(is_expression auto a, is_expression auto b) {
  return Tree(PRODUCT, bind(a), bind(b));
}

constexpr auto operator-(is_expression auto a, is_expression auto b) {
  return Tree(DIFFERENCE, bind(a), bind(b));
}

constexpr auto operator-(is_expression auto a) {
  return Rational(-1) * bind(a);
}

constexpr auto operator/(is_expression auto a, is_expression auto b) {
  return Tree(INVERSE, bind(a), bind(b));
}

constexpr auto D(is_expression auto a, Index i, std::same_as<Index> auto... is) {
  Index j = (i + ... + is);
  return Tree(PARTIAL, bind(a), Tree(j));
}

constexpr auto delta(Index a, Index b) {
  Index ab = a + b;
  assert(ab.size() == 2);
  return Tree(ab);
}

constexpr auto symmetrize(is_expression auto a) {
  Tree t = bind(a);
  Index i = t.outer();
  return Rational(1,2) * (t + t(reverse(i)));
}

constexpr Tree<3> Tensor::operator()(std::same_as<Index> auto... is) const {
  Index i = (is + ... + Index{});
  assert(i.size() == order_);
  return Tree(BIND, Tree(*this), Tree(i));
}
}

template <>
struct fmt::formatter<ttl::Tag> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::Tag& tag, FormatContext& ctx) {
    switch (tag) {
     case ttl::SUM:        return format_to(ctx.out(), "{}", "+");
     case ttl::DIFFERENCE: return format_to(ctx.out(), "{}", "-");
     case ttl::PRODUCT:    return format_to(ctx.out(), "{}", "*");
     case ttl::INVERSE:    return format_to(ctx.out(), "{}", "/");
     case ttl::BIND:       return format_to(ctx.out(), "{}", "()");
     case ttl::PARTIAL:    return format_to(ctx.out(), "{}", "dx");
     case ttl::INDEX:      return format_to(ctx.out(), "{}", "index");
     case ttl::TENSOR:     return format_to(ctx.out(), "{}", "tensor");
     case ttl::RATIONAL:   return format_to(ctx.out(), "{}", "q");
     case ttl::DOUBLE:     return format_to(ctx.out(), "{}", "d");
     default:
      __builtin_unreachable();
    }
  }
};

template <>
struct fmt::formatter<ttl::Node> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::Node& node, FormatContext& ctx) {
    switch (node.tag_) {
     case ttl::SUM:
     case ttl::DIFFERENCE:
     case ttl::PRODUCT:
     case ttl::INVERSE:
     case ttl::BIND:
     case ttl::PARTIAL:    return format_to(ctx.out(), "{}", node.tag_);
     case ttl::INDEX:      return format_to(ctx.out(), "{}", node.index_.to_string());
     case ttl::TENSOR:     return format_to(ctx.out(), "{}", node.tensor_.id());
     case ttl::RATIONAL:   return format_to(ctx.out(), "{}", node.q_.to_string());
     case ttl::DOUBLE:     return format_to(ctx.out(), "{}", std::to_string(node.d_));
     default:
      __builtin_unreachable();
    }
  }
};

template <int M>
struct fmt::formatter<ttl::Tree<M>>
{
  static constexpr const char dot_fmt[] = "dot";
  static constexpr const char eqn_fmt[] = "eqn";

  bool dot_ = false;
  bool eqn_ = true;

  constexpr auto parse(format_parse_context& ctx) {
    auto i = ctx.begin(), e = ctx.end();
    if (i == e) {
      return i;
    }

    i = std::strchr(i, '}');
    if (i == ctx.begin()) {
      return i;
    }

    if (i == e) {
      throw fmt::format_error("invalid format");
    }

    if ((dot_ = std::equal(ctx.begin(), i, std::begin(dot_fmt)))) {
      eqn_ = false;
      return i;
    }

    if ((eqn_ = std::equal(ctx.begin(), i, std::begin(eqn_fmt)))) {
      return i;
    }

    throw fmt::format_error("invalid format");
  }

  auto format(const ttl::Tree<M>& a, auto& ctx) {
    assert(dot_^ eqn_);
    return (dot_) ? dot(a, ctx) : eqn(a, ctx);
  }

 private:
  auto dot(const ttl::Tree<M>& a, auto& ctx) const {
    for (int i = 0; const auto& node : a.nodes) {
      format_to(ctx.out(), FMT_STRING("\tnode{}[label=\"{}\"]\n"), i, node);
      if (node.is_binary()) {
        format_to(ctx.out(), FMT_STRING("\tnode{} -- node{}\n"), i, a.left[i]);
        format_to(ctx.out(), FMT_STRING("\tnode{} -- node{}\n"), i, i - 1);
      }
      ++i;
    }
    return ctx.out();
  }

  auto eqn(const ttl::Tree<M>& a, auto& ctx) const {
    auto str = a.postfix(
        [](auto leaf) {
          return fmt::format(FMT_STRING("{}"), leaf);
        },
        [](auto node, auto l, auto r) {
          if (node.is(ttl::PARTIAL)) {
            return fmt::format(FMT_STRING("D({},{})"), l, r);
          }
          if (node.is(ttl::BIND)) {
            return fmt::format(FMT_STRING("{}({})"), l, r);
          }
          return fmt::format(FMT_STRING("({} {} {})"), l, node, r);
        });
    return format_to(ctx.out(), FMT_STRING("{}"), str);
  }
};
