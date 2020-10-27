#pragma once

#include "Index.hpp"
#include "Tensor.hpp"
#include "concepts.hpp"
#include "utils.hpp"

namespace ttl {
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

template <Tag> struct tag_t {};
template <Tag tag> constexpr inline tag_t<tag> tag_v = {};

constexpr bool is_binary(Tag tag) {
  return tag < INDEX;
}

union Node {
  Index     index;
  Tensor   tensor;
  Rational      q;
  double        d;

  constexpr Node()              noexcept : index() {}
  constexpr Node(Index index)   noexcept : index(index) {}
  constexpr Node(Tensor tensor) noexcept : tensor(tensor) {}
  constexpr Node(Rational q)    noexcept : q(q) {}
  constexpr Node(double d)      noexcept : d(d) {}
};

template <typename T> requires(std::same_as<std::remove_cv_t<T>, Node>)
struct TaggedNode {
  Tag tag;
  T& node;
  constexpr TaggedNode(Tag tag, T& node) : tag(tag), node(node) {}

  template <Tag tag>
  constexpr TaggedNode(tag_t<tag>, T& node) : tag(tag), node(node) {}

  constexpr bool is(Tag t) const {
    return t == tag;
  }

  constexpr bool is_binary() const {
    return ttl::is_binary(tag);
  }

  constexpr decltype(auto) index() const {
    return (tag < TENSOR) ? &node.index : nullptr;
  }

  constexpr decltype(auto) tensor() const {
    return (tag == TENSOR) ? &node.tensor : nullptr;
  }

  constexpr decltype(auto) rational() const {
    return (tag == TENSOR) ? &node.q : nullptr;
  }

  constexpr decltype(auto) value() const {
    return (tag == TENSOR) ? &node.d : nullptr;
  }
};

template <Tag... Ts>
struct TaggedTree {
  constexpr static std::true_type is_tree_tag = {};
  constexpr static int M = sizeof...(Ts);
  constexpr static std::array<Tag, M> tags = { Ts... };

  std::array<Node, M> nodes;

  constexpr TaggedTree()              noexcept = default;
  constexpr TaggedTree(Tensor tensor) noexcept : nodes { tensor } {}
  constexpr TaggedTree(Index index)   noexcept : nodes { index }  {}
  constexpr TaggedTree(Rational q)    noexcept : nodes { q }      {}
  constexpr TaggedTree(double d)      noexcept : nodes { d }      {}

  template <Tag... As, Tag... Bs, Tag C>
  requires(C < INDEX)
  constexpr TaggedTree(TaggedTree<As...> a, TaggedTree<Bs...> b, tag_t<C>) {
    int i = 0;
    for (auto&& a : a.nodes) nodes[i++] = a;
    for (auto&& b : b.nodes) nodes[i++] = b;

    Index ai = a.outer(), bi = b.outer();
    switch (C) {
     case SUM:
     case DIFFERENCE:
      assert(permutation(ai, bi));
      nodes[i++].index = ai;
      break;
     case PRODUCT:
     case INVERSE:
      nodes[i++].index = ai ^ bi;
      break;
     case BIND:
     case PARTIAL:
      nodes[i++].index = exclusive(ai + bi);
      break;
     default:
      __builtin_unreachable();
    };
  }

  constexpr static Tag tag(int i) {
    return tags[M - i - 1];
  }

  constexpr const Node& node(int i) const {
    return nodes[i];
  }

  constexpr Node& node(int i) {
    return nodes[i];
  }

  constexpr TaggedNode<const Node> at(int i) const {
    return { tag(i), node(i) };
  }

  constexpr TaggedNode<Node> at(int i) {
    return { tag(i), node(i) };
  }

  constexpr TaggedNode<const Node> root() const {
    return { tag(M - 1), node(M - 1) };
  }

  constexpr Index outer() const {
    if (const Index* i = root().index()) {
      return *i;
    }
    return {};
  }

  constexpr void rewrite(Index replace)
  {
    // Rewrite the outer index of a tree.
    Index search = outer();
    assert(replace.size() == search.size());
    for (int i = 0; i < nodes.size(); ++i) {
      if (Index *idx = at(i).index()) {
        idx->search_and_replace(search, replace);
      }
    }
  }

  constexpr TaggedTree operator()(std::same_as<Index> auto... is) const {
    TaggedTree copy(*this);
    copy.rewrite((Index{} + ... + is));
    return copy;
  }

  constexpr friend auto postfix(const TaggedTree& tree, auto&&... ops) {
    utils::overloaded op = { ops... };
    using T = decltype(op(tree.at(0)));
    ce::cvector<T, TaggedTree::M> stack;
    for (int i = 0; i < M; ++i) {
      auto&& node = tree.at(i);
      if (node.is_binary()) {
        auto&& r = stack.pop_back();
        auto&& l = stack.pop_back();
        stack.push_back(op(std::move(node), std::move(l), std::move(r)));
      }
      else {
        stack.push_back(op(std::move(node)));
      }
    }
    assert(stack.size() == 1);
    return stack.pop_back();
  }
};

TaggedTree(Tensor)   -> TaggedTree<TENSOR>;
TaggedTree(Index)    -> TaggedTree<INDEX>;
TaggedTree(Rational) -> TaggedTree<RATIONAL>;
TaggedTree(double)   -> TaggedTree<DOUBLE>;

template <Tag... As, Tag... Bs, Tag C>
requires(C < INDEX)
TaggedTree(TaggedTree<As...>, TaggedTree<Bs...>, tag_t<C>) ->
  TaggedTree<C, Bs..., As...>;

template <typename T>
concept is_expression =
 is_tree<T> ||
 std::same_as<T, Tensor> ||
 std::same_as<T, Index> ||
 std::same_as<T, Rational> ||
 std::signed_integral<T> ||
 std::same_as<T, double>;

constexpr auto bind(is_expression auto a) {
  return TaggedTree(a);
}

constexpr auto bind(std::signed_integral auto t) {
  return bind(Rational(t));
}

constexpr auto bind(is_tree auto a) {
  return a;
}

constexpr auto operator+(is_expression auto a) {
  return bind(a);
}

constexpr auto operator+(is_expression auto a, is_expression auto b) {
  return TaggedTree(bind(a), bind(b), tag_v<SUM>);
}

constexpr auto operator*(is_expression auto a, is_expression auto b) {
  return TaggedTree(bind(a), bind(b), tag_v<PRODUCT>);
}

constexpr auto operator-(is_expression auto a, is_expression auto b) {
  return TaggedTree(bind(a), bind(b), tag_v<DIFFERENCE>);
}

constexpr auto operator-(is_expression auto a) {
  return Rational(-1) * bind(a);
}

constexpr auto operator/(is_expression auto a, is_expression auto b) {
  return TaggedTree(bind(a), bind(b), tag_v<INVERSE>);
}

constexpr auto D(is_expression auto a, Index i, std::same_as<Index> auto... is) {
  return TaggedTree(bind(a), bind((i + ... + is)), tag_v<PARTIAL>);
}

constexpr auto delta(Index a, Index b) {
  assert(a.size() == 1);
  assert(b.size() == 1);
  assert(a != b);
  return bind((a + b));
}

constexpr auto symmetrize(is_expression auto a) {
  TaggedTree t = bind(a);
  Index i = t.outer();
  return Rational(1,2) * (t + t(reverse(i)));
}

constexpr auto
Tensor::operator()(std::same_as<Index> auto... is) const {
  Index i = (is + ... + Index{});
  assert(i.size() == order_);
  return TaggedTree(bind(*this), bind(i), tag_v<BIND>);
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

template <typename T>
struct fmt::formatter<ttl::TaggedNode<T>> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::TaggedNode<T>& node, FormatContext& ctx) {
    switch (node.tag) {
     case ttl::SUM:
     case ttl::DIFFERENCE:
     case ttl::PRODUCT:
     case ttl::INVERSE:
     case ttl::BIND:
     case ttl::PARTIAL:    return format_to(ctx.out(), "{}", node.tag);
     case ttl::INDEX:      return format_to(ctx.out(), "{}", node.node.index);
     case ttl::TENSOR:     return format_to(ctx.out(), "{}", node.node.tensor);
     case ttl::RATIONAL:   return format_to(ctx.out(), "{}", node.node.q);
     case ttl::DOUBLE:     return format_to(ctx.out(), "{}", node.node.d);
     default:
      __builtin_unreachable();
    }
  }
};

template <ttl::Tag... Ts>
struct fmt::formatter<ttl::TaggedTree<Ts...>>
{
  using Tree = ttl::TaggedTree<Ts...>;
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

  auto format(const Tree& a, auto& ctx) {
    assert(dot_^ eqn_);
    return (dot_) ? dot(a, ctx) : eqn(a, ctx);
  }

 private:
  auto dot(const Tree& a, auto& ctx) const {
    for (int i = 0; i < Tree::M; ++i) {
      format_to(ctx.out(), FMT_STRING("\tnode{}[label=\"{}\"]\n"), i, a.at(i));
    }

    int i = 0;
    postfix(a,
            [&](auto&&) { return i++; },
            [&](auto&&, int l, int r) {
              format_to(ctx.out(), FMT_STRING("\tnode{} -- node{}\n"), i, l);
              format_to(ctx.out(), FMT_STRING("\tnode{} -- node{}\n"), i, r);
              return i++;
            });
    return ctx.out();
  }

  auto eqn(const Tree& a, auto& ctx) const {
    auto str = postfix(
        a,
        [](auto&& leaf) {
          return fmt::format(FMT_STRING("{}"), std::move(leaf));
        },
        [](auto&& node, auto&& l, auto&& r) {
          if (node.is(ttl::PARTIAL)) {
            return fmt::format(FMT_STRING("D({},{})"), std::move(l), std::move(r));
          }
          if (node.is(ttl::BIND)) {
            return fmt::format(FMT_STRING("{}({})"), std::move(l), std::move(r));
          }
          return fmt::format(FMT_STRING("({} {} {})"), std::move(l), std::move(node), std::move(r));
        });
    return format_to(ctx.out(), FMT_STRING("{}"), str);
  }
};
