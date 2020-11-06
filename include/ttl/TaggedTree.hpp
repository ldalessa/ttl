#pragma once

#include "TaggedNode.hpp"
#include "concepts.hpp"
#include "utils.hpp"

namespace ttl {
template <Tag... Ts> requires(sizeof...(Ts) != 0)
struct TaggedTree {
  constexpr static std::true_type is_tree_tag = {};
  constexpr static int M = sizeof...(Ts);
  constexpr static Tag tags[M] = { Ts... };

  Node nodes_[M];
  int  depth_ = 1;

  /// Leaf tree construction.
  ///
  /// There are CTAD guides that infer the right tag type for each of these
  /// four leaf operations.
  constexpr TaggedTree(Tensor tensor) noexcept : nodes_ { tensor } {}
  constexpr TaggedTree(Index index)   noexcept : nodes_ { index }  {}
  constexpr TaggedTree(Rational q)    noexcept : nodes_ { q }      {}
  constexpr TaggedTree(double d)      noexcept : nodes_ { d }      {}

  /// Split constructor.
  ///
  /// This mess just says that we can create a TaggedTree from a sequence of
  /// objects, as long as there are as many objects as there are tags, and each
  /// passed is actually a Node. It's used when we're decomposing trees.
  template <typename... Nodes>
  requires((sizeof...(Nodes) == sizeof...(Ts)) &&
           (std::same_as<Node, std::remove_cvref_t<Nodes>> && ...))
  constexpr TaggedTree(int depth, Nodes&&... nodes)
    : nodes_ { std::forward<Nodes>(nodes)... }
    , depth_ { depth }
  {
  }

  /// Join constructor.
  ///
  /// This constructor joins two trees with a binary node with tag C.
  template <Tag... As, Tag... Bs, Tag C>
  requires(C < INDEX)
  constexpr TaggedTree(TaggedTree<As...> a, TaggedTree<Bs...> b, tag_t<C>)
    : depth_ { std::max(a.depth(), b.depth()) + 1 }
  {
    // if the left child is a delta expression, then it's parent should not be a
    // product (this is a tree structure invariant we enforce, not a constraint
    // on the syntax)
    assert(!a.root().is(DELTA) || (C != PRODUCT));

    // an index must be the right child of a BIND or PARTIAL node
    assert(!a.root().is(INDEX));
    assert(!b.root().is(INDEX) || (C == BIND || C == PARTIAL));

    int i = 0;
    for (auto&& a : a.nodes_) nodes_[i++] = a;
    for (auto&& b : b.nodes_) nodes_[i++] = b;

    Index ai = a.outer(), bi = b.outer();
    switch (C) {
     case SUM:
     case DIFFERENCE:
      assert(permutation(ai, bi));
      nodes_[i++].index = ai;
      break;
     case PRODUCT:
     case INVERSE:
      nodes_[i++].index = ai ^ bi;
      break;
     case BIND:
     case PARTIAL:
      nodes_[i++].index = exclusive(ai + bi);
      break;
     default:
      __builtin_unreachable();
    };
  }

  constexpr friend int size(const TaggedTree&) {
    return M;
  }

  constexpr int depth() const {
    return depth_;
  }

  constexpr static int size() {
    return M;
  }

  constexpr static int n_tensors() {
    return ((Ts == TENSOR) + ... + 0);
  }

  constexpr static Tag tag(int i) {
    return tags[M - i - 1];
  }

  constexpr const Node& node(int i) const {
    return nodes_[i];
  }

  constexpr Node& node(int i) {
    return nodes_[i];
  }

  constexpr TaggedNode<const Node> at(int i) const {
    return { i, tag(i), node(i) };
  }

  constexpr TaggedNode<Node> at(int i) {
    return { i, tag(i), node(i) };
  }

  constexpr TaggedNode<const Node> root() const {
    return { M - 1, tag(M - 1), node(M - 1) };
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
    for (int i = 0; i < M; ++i) {
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

  constexpr auto split() const {
    // postorder traversal to find the split point
    constexpr int nl = [] {
      int l = 0;
      utils::stack<int> stack;
      for (int i = 0; i < M; ++i) {
        if (is_binary(tag(i))) {
          int r = stack.pop();
          assert(r == i - 1);
          l = stack.pop();
        }
        stack.push(i);
      }
      return l + 1;
    }();

    constexpr int nr = M - nl - 1;

    // create the left and right child trees (the tags are stored backwards, and
    // we need to preserve that order in the children types, so we need a little
    // bit of fanciness to make sure we're picking up the right ones)
    auto left = [&]<std::size_t... i>(std::index_sequence<i...>) {
      return TaggedTree<tags[M - nl + i]...>(depth() - 1, nodes_[i]...);
    }(std::make_index_sequence<nl>());

    auto right = [&]<std::size_t... i>(std::index_sequence<i...>) {
      return TaggedTree<tags[M - nl - nr + i]...>(depth() - 1, nodes_[nl + i]...);
    }(std::make_index_sequence<nr>());

    // return the pair
    return std::tuple(tag(M - 1), left, right);
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

template <typename A> requires(is_expression<A>)
constexpr auto operator*(A a, is_expression auto b) {
  // tree invariant is that delta nodes must appear as right children
  if constexpr (!is_tree<A>) {
    return TaggedTree(bind(a), bind(b), tag_v<PRODUCT>);
  }
  else if constexpr (A::tag(A::size() - 1) != DELTA) {
    return TaggedTree(bind(a), bind(b), tag_v<PRODUCT>);
  }
  else {
    return TaggedTree(bind(b), bind(a), tag_v<PRODUCT>); // commute
  }
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
  return TaggedTree<DELTA>(a + b);
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
    ttl::utils::stack<int> stack;
    for (int i = 0; i < a.size(); ++i) {
      ttl::TaggedNode node = a.at(i);
      if (node.is_binary()) {
        format_to(ctx.out(), "\tnode{}[label=\"{} <{}>\"]\n", i, node, *node.index());
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, stack.pop());
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, stack.pop());
      }
      else {
        format_to(ctx.out(), "\tnode{}[label=\"{}\"]\n", i, a.at(i));
      }
      stack.push(i);
    }
    return ctx.out();
  }

  auto eqn(const Tree& a, auto& ctx) const {
    ttl::utils::stack<std::string> stack;
    for (int i = 0; i < a.size(); ++i) {
      ttl::TaggedNode node = a.at(i);
      if (node.is(ttl::PARTIAL)) {
        stack.push(fmt::format("D({1},{0})", stack.pop(), stack.pop()));
      }
      else if (node.is(ttl::BIND)) {
        stack.push(fmt::format("{1}({0})", stack.pop(), stack.pop()));
      }
      else if (node.is_binary()) {
        stack.push(fmt::format("({1} {2} {0})", stack.pop(), stack.pop(), node));
      }
      else if (node.is(ttl::DELTA)) {
        stack.push(fmt::format("delta({})", node));
      }
      else {
        stack.push(fmt::to_string(node));
      }
    }
    return format_to(ctx.out(), "{}", stack.pop());
  }
};
