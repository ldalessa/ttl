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
  PARTIAL,  // last binary
  INDEX,    // last index state
  TENSOR,
  RATIONAL,
  DOUBLE
};

// Tag queries.
constexpr bool      is_binary(Tag tag) { return tag < INDEX; }
constexpr bool        is_leaf(Tag tag) { return !is_binary(tag); }
constexpr bool is_index_valid(Tag tag) { return tag < TENSOR; }

union State {
  struct {} _ = {};                             // default state
  Index index;
  Tensor tensor;
  Rational q;
  double d;
};

constexpr auto visit(Tag tag, auto&& state, auto&& op) {
  switch (tag) {
   case SUM:
   case DIFFERENCE:
   case PRODUCT:
   case INVERSE:
   case BIND:
   case PARTIAL:
   case INDEX:    return op(state.index);
   case TENSOR:   return op(state.tensor);
   case RATIONAL: return op(state.q);
   case DOUBLE:   return op(state.d);
  }
  __builtin_unreachable();
}

template <int M = 1>
struct Tree
{
  constexpr friend std::true_type is_tree_v(Tree) { return {}; }

  ce::cvector<int, M> left;
  ce::cvector<Tag, M> tags;
  ce::cvector<State, M> state;

  constexpr Tree() = default;
  constexpr Tree(const Tree&) = default;
  constexpr Tree(Tree&&) = default;
  constexpr Tree& operator=(const Tree&) = default;
  constexpr Tree& operator=(Tree&&) = default;

  constexpr Tree(Tensor tensor)
      :  left(std::in_place, -1)
      ,  tags(std::in_place, TENSOR)
      , state(std::in_place, State{.tensor = tensor})
  {
  }

  constexpr Tree(Index index)
      :  left(std::in_place, -1)
      ,  tags(std::in_place, INDEX)
      , state(std::in_place, State{.index = index})
  {
  }

  constexpr Tree(Rational q)
      :  left(std::in_place, -1)
      ,  tags(std::in_place, RATIONAL)
      , state(std::in_place, State{.q = q})
  {
  }

  constexpr Tree(double d)
      :  left(std::in_place, -1)
      ,  tags(std::in_place, DOUBLE)
      , state(std::in_place, State{.d = d})
  {
  }

  template <int A, int B>
  constexpr Tree(Tag tag, Tree<A> a, Tree<B> b)
  {
    for (int i = 0; i < a.size(); ++i) {
      left.push_back(a.left[i]);
      tags.push_back(a.tags[i]);
      state.push_back(a.state[i]);
    }
    for (int i = 0; i < b.size(); ++i) {
      left.push_back(b.left[i] + a.size());
      tags.push_back(b.tags[i]);
      state.push_back(b.state[i]);
    }
    left.push_back(a.size() - 1);
    tags.push_back(tag);

    Index l = a.outer();
    Index r = b.outer();
    switch (tag) {
     case SUM:
     case DIFFERENCE:
      assert(permutation(l, r));
      state.push_back({.index = l});
      break;
     case PRODUCT:
     case INVERSE:
      state.push_back({.index = l ^ r});
      break;
     case BIND:
     case PARTIAL:
      state.push_back({.index = exclusive(l + r)});
      break;
     default: __builtin_unreachable();
    }
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
    for (int i = 0; i < size(); ++i) {
      if (is_index_valid(tags[i])) {
        state[i].index.search_and_replace(search, replace);
      }
    }
  }

  constexpr Tree operator()(Index i, std::same_as<Index> auto... is) const {
    Tree copy(*this);
    copy.rewrite((i + ... + is));
    return copy;
  }

  constexpr Index outer() const {
    if (is_index_valid(tags.back())) {
      return state.back().index;
    }
    return {};
  }

  constexpr auto visit(const auto& op, int i) const {
    switch (tags[i]) {
     case TENSOR:   return op(state[i].tensor);
     case INDEX:    return op(state[i].index);
     case RATIONAL: return op(state[i].q);
     case DOUBLE:   return op(state[i].d);
     default: return op(tags[i], visit(op, left[i]), visit(op, i - 1));
    };
    __builtin_unreachable();
  }

  constexpr auto visit(const auto&... ops) const {
    return visit(overloaded { ops...}, size() - 1);
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

constexpr auto bind(Tensor t) {
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
  auto format(const ttl::Tag& tag, FormatContext& ctx) {
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
      return ctx.out();
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
    if (dot_) {
      return dot(a, ctx);
    }
    return format_to(ctx.out(), "{}", eqn(a));
  }

 private:
  auto dot(const ttl::Tree<M>& a, auto& ctx) const {
    for (int i = 0; i < a.size(); ++i) {
      ttl::Tag tag = a.tags[i];
      if (ttl::is_binary(tag)) {
        format_to(ctx.out(), "\tnode{}[label=\"{}\"]\n", i, tag);
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, a.left[i]);
        format_to(ctx.out(), "\tnode{} -- node{}\n", i, i - 1);
      }
      else {
        std::string label = visit(tag, a.state[i], [](auto a) {
          return to_string(a);
        });
        format_to(ctx.out(), "\tnode{}[label=\"{}\"]\n", i, label);
      }
    }
    return ctx.out();
  }

  std::string eqn(const ttl::Tree<M>& a) const {
    return a.visit(
        [](auto leaf) {
          return fmt::format("{}", leaf);
        },
        [](ttl::Tag tag, auto l, auto r) {
          switch (tag) {
           case ttl::SUM:
           case ttl::PRODUCT:
           case ttl::DIFFERENCE:
           case ttl::INVERSE:
            return fmt::format("({} {} {})", l, tag, r);
           case ttl::PARTIAL:
            return fmt::format("D({},{})", l, r);
           case ttl::BIND:
            if (r.size()) {
              return fmt::format("{}({})", l, r);
            }
            else {
              return fmt::format("{}", l);
            }
           default:
            __builtin_unreachable();
          };
        });
  }
};
