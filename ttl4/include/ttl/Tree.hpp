#pragma once

#include "Rational.hpp"
#include "Tensor.hpp"
#include "mp/cvector.hpp"
#include "mp/variant_utils.hpp"
#include <concepts>
#include <functional>
#include <iostream>
#include <utility>
#include <variant>

namespace ttl
{
struct Product {
  constexpr friend Index outer(const Product&, Index a, Index b) {
    return a ^ b;
  }

  friend std::ostream& operator<<(std::ostream& os, const Product&) {
    return os << "*";
  }
};

struct Sum {
  constexpr friend Index outer(const Sum&, Index a, Index b) {
    assert(permutation(a, b));
    return a;
  }

  friend std::ostream& operator<<(std::ostream& os, const Sum&) {
    return os << "+";
  }
};

struct Inverse {
  constexpr friend Index outer(const Inverse&, Index a, Index b) {
    return a ^ b;
  }

  friend std::ostream& operator<<(std::ostream& os, const Inverse&) {
    return os << "/";
  }
};

class Partial {
  Index dx_;

 public:
  constexpr Partial(Index dx) : dx_(dx) {
  }

  constexpr Index outer(Index a) const {
    return a ^ dx_;
  }

  friend std::ostream& operator<<(std::ostream& os, const Partial& dx) {
    return os << "dx(" << dx.dx_ << ")";
  }
};

constexpr Index outer(const Partial& dx, Index a) {
  return dx.outer(a);
}

class Bind {
  Index outer_;

 public:
  constexpr Bind(Index outer) : outer_(outer) {
  }

  constexpr Index outer() const {
    return exclusive(outer_);
  }

  constexpr Index inner() const {
    return repeated(outer_);
  }

  constexpr Index all() const {
    return outer() + inner();
  }

  friend std::ostream& operator<<(std::ostream& os, const Bind& bind) {
    return os << "bind(" << bind.all() << ")";
  }
};

constexpr Index outer(const Bind& bind, Index) {
  return bind.outer();
}

class Delta {
  Index outer_;

 public:
  constexpr Delta(Index outer) : outer_(outer) {
    assert(outer.size() == 2);
  }

  constexpr friend Index outer(const Delta& delta) {
    return delta.outer_;
  }

  friend std::ostream& operator<<(std::ostream& os, const Delta& delta) {
    return os << "delta(" << delta.outer_ << ")";
  }
};

using TensorRef = std::reference_wrapper<const Tensor>;

template <typename T>
concept Leaf =
 std::same_as<Delta,     std::remove_cvref_t<T>> ||
 std::same_as<TensorRef, std::remove_cvref_t<T>> ||
 std::same_as<Rational,  std::remove_cvref_t<T>> ||
 std::same_as<double,    std::remove_cvref_t<T>>;

template <typename T>
concept Unary =
 std::same_as<Bind,    std::remove_cvref_t<T>> ||
 std::same_as<Partial, std::remove_cvref_t<T>>;

template <typename T>
concept Binary =
 std::same_as<Product, std::remove_cvref_t<T>> ||
 std::same_as<Sum,     std::remove_cvref_t<T>> ||
 std::same_as<Inverse, std::remove_cvref_t<T>>;

constexpr Index outer(const Tensor&)   { return {}; }
constexpr Index outer(const Rational&) { return {}; }
constexpr Index outer(const double&)   { return {}; }

using Node = std::variant<Bind, Product, Sum, Inverse, Partial, Delta, TensorRef, Rational, double>;

template <int M>
class Tree
{
  Node nodes_[M] = {};

  template <int A, int B, std::size_t... As, std::size_t... Bs>
  constexpr Tree(const Tree<A>& a, const Tree<B>& b, Binary auto c,
                 std::index_sequence<As...>, std::index_sequence<Bs...>)
      : nodes_ { a[As]..., b[Bs]..., c }
  {
  }

  template <int A, std::size_t... As>
  constexpr Tree(const Tree<A>& a, Unary auto b, std::index_sequence<As...>)
      : nodes_ { a[As]...,  b }
  {
  }

 public:
  constexpr Tree() = delete;
  constexpr Tree(const Tree&) = delete;
  constexpr Tree(Tree&&) = delete;

  /// Join two trees with a binary node.
  template <int A, int B>
  constexpr Tree(const Tree<A>& a, const Tree<B>& b, Binary auto c)
      : Tree(a, b, c,
             std::make_index_sequence<A>(),
             std::make_index_sequence<B>())
  {
  }

  /// Wrap a tree in a unary node.
  template <int A>
  constexpr Tree(const Tree<A>& a, Unary auto b)
      : Tree(a, b, std::make_index_sequence<A>())
  {
  }

  /// Construct a tree from a leaf node.
  constexpr Tree(Leaf auto leaf)
      : nodes_ {  leaf }
  {
  }

  constexpr int size() const {
    return M;
  }

  constexpr auto begin() const { return nodes_; }
  constexpr auto   end() const { return nodes_ + M; }

  /// Access to the underlying data.
  constexpr decltype(auto) operator[](int i) const {
    assert(0 <= i && i < M);
    return nodes_[i];
  }

  /// Rebind the index in a tree.
  constexpr Tree<M + 1> operator()(auto... is) const {
    return Tree<M + 1>(*this, Bind((is + ... + Index())));
  }
};

template <int A, int B>
Tree(const Tree<A>&, const Tree<B>&, Binary auto) -> Tree<A + B + 1>;

template <int A>
Tree(const Tree<A>&, Unary auto) -> Tree<A + 1>;

Tree(Leaf auto) -> Tree<1>;

/// This utility visits the tree bottom-up.
///
/// The user must provide an Op that handles each type of node and returns and
/// intermediate state that is requires. This state will be passed back into the
/// Op with the parent node.
///
/// ```
///   Op<int, Binary, State, State> -> State
///   Op<int, Unary, State> -> State
///   Op<int, Leaf> -> State
/// ```
///
/// @tparam           M The size of the tree.
/// @tparam          Op The node handler type.
///
/// @param         tree The tree to traverse.
/// @param           op The node handler.
///
/// @returns            The value returned by the root node in the tree.
template <int M, typename Op>
constexpr auto visit(const Tree<M>& tree, Op&& op) {
  using State = decltype(op(0, 1.0));
  mp::cvector<State, M> stack;
  for (int i = 0; const Node& n : tree) {
    std::visit([&](const auto& node) {
      if constexpr (Binary<decltype(node)>) {
        auto b = stack.pop();
        auto a = stack.pop();                   // could use back()
        stack.push(op(i, node, a, b));
      }
      else if constexpr (Unary<decltype(node)>) {
        auto a = stack.pop();                   // could use back()
        stack.push(op(i, node, a));
      }
      else {
        assert(Leaf<decltype(node)>);
        stack.push(op(i, node));
      }
    }, n);
    ++i;
  }
  assert(stack.size() == 1);
  return stack.pop();
}

/// We need to execute the tree in order to know what its outer index is
/// supposed to be.
template <int M>
constexpr Index outer(const Tree<M>& tree) {
  return visit(tree, [&](int, const auto& node, auto&&... args) {
    return outer(node, std::forward<decltype(args)>(args)...);
  });
}

template <typename> constexpr inline bool is_tree_v = false;
template <int M>    constexpr inline bool is_tree_v<Tree<M>> = true;

template <typename T>
concept Expression =
 is_tree_v<std::remove_cvref_t<T>> ||
 Leaf<T> ||
 std::same_as<const Tensor&, T> ||
 std::signed_integral<T>;

constexpr Tree<2> Tensor::operator()(auto... is) const {
  return Tree(Tree(std::cref(*this)), Bind((is + ... + Index())));
}

constexpr auto bind(const Tensor& t) {
  assert(t.order() == 0);
  return t();
}

template <int M>
constexpr decltype(auto) bind(const Tree<M>& a) {
  return a;
}

template <int M>
constexpr decltype(auto) bind(Tree<M>&& a) {
  return std::move(a);
}

template <Leaf A>
constexpr decltype(auto) bind(A&& a) {
  return Tree(std::forward<A>(a));
}

constexpr auto bind(std::signed_integral auto v) {
  return bind(Rational(v));
}

template <Expression A>
constexpr auto operator+(A&& a) {
  return bind(std::forward<A>(a));
}

template <Expression A, Expression B>
constexpr auto operator+(A&& a, B&& b) {
  return Tree(bind(std::forward<A>(a)), bind(std::forward<B>(b)), Sum());
}

template <Expression A, Expression B>
constexpr auto operator*(A&& a, B&& b) {
  return Tree(bind(std::forward<A>(a)), bind(std::forward<B>(b)), Product());
}

template <Expression A, Expression B>
constexpr auto operator/(A&& a, B&& b) {
  return Tree(bind(std::forward<A>(a)), bind(std::forward<B>(b)), Inverse());
}

template <Expression A>
constexpr auto operator-(A&& a) {
  return Rational(-1) * bind(std::forward<A>(a));
}

template <Expression A, Expression B>
constexpr auto operator-(A&& a, B&& b) {
  return bind(std::forward<A>(a)) + (-std::forward<B>(b));
}

template <Expression A>
constexpr auto D(A&& a, auto... is) {
  return Tree(bind(std::forward<A>(a)), Partial((is + ... + Index())));
}

constexpr auto delta(Index a, Index b) {
  assert(a.size() == 1);
  assert(b.size() == 1);
  return bind(Delta(a + b));
}

constexpr auto symmetrize(Expression auto&& a) {
  return Rational(1,2) * (a + a(reverse(outer(a))));
}
}
