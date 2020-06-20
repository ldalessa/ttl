#pragma once

#include "Rational.hpp"
#include "Tensor.hpp"
#include "mp/variant_utils.hpp"
#include <concepts>
#include <functional>
#include <utility>
#include <variant>

namespace ttl
{

class Common
{
 protected:
  Index outer_;

  constexpr Common(Index outer) : outer_(outer) {
  }

 public:

  constexpr Index outer() const {
    return outer_;
  }

};

class Product : public Common {
 public:
  constexpr Product(Index a, Index b) : Common(a ^ b) {
  }
};

class Sum : public Common {
 public:
  constexpr Sum(Index a, Index b) : Common(a) {
    assert(permutation(a, b));
  }
};

class Inverse : public Common {
 public:
  constexpr Inverse(Index a, Index b) : Common(a ^ b) {
  }
};

class Partial : public Common {
  Index dx_;

 public:
  constexpr Partial(Index a, Index dx) : Common(exclusive(a + dx)), dx_(dx) {
  }

  constexpr Index dx() const {
    return dx_;
  }
};

class Bind : public Common {
 public:
  constexpr Bind(Index bound) : Common(bound) {
  }
};

class Delta : public Common {
 public:
  constexpr Delta(Index outer) : Common(outer) {
    assert(outer.size() == 2);
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

constexpr Index outer(const Common& node) { return node.outer(); }
constexpr Index outer(const Tensor&)      { return {}; }
constexpr Index outer(const Rational&)    { return {}; }
constexpr Index outer(const double&)      { return {}; }

using Node = std::variant<Bind, Product, Sum, Inverse, Partial, Delta, TensorRef, Rational, double>;

template <int M>
class Tree
{
  template <int> friend class Tree;

  Node nodes_[M] = {};
  int   left_[M] = {};

  template <int A, int B, std::size_t... As, std::size_t... Bs>
  constexpr Tree(const Tree<A>& a, const Tree<B>& b, Binary auto c,
                 std::index_sequence<As...>, std::index_sequence<Bs...>)
      : nodes_ { a.nodes_[As]..., b.nodes_[Bs]..., c }
      , left_  { a.left_[As]..., (b.left_[Bs] + A)..., A - 1 }
  {
  }

  template <int A, std::size_t... As>
  constexpr Tree(const Tree<A>& a, Unary auto b, std::index_sequence<As...>)
      : nodes_ { a.nodes_[As]...,  b }
      , left_  { a.left_[As]...,  -1 }
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
      , left_  { -1 }
  {
  }

  constexpr int size() const {
    return M;
  }

  constexpr auto begin() const { return nodes_; }
  constexpr auto   end() const { return nodes_ + M; }

  constexpr int index_of(const Node& node) const {
    int i = &node - nodes_;
    assert(0 <= i && i < M);
    return i;
  }

  template <mp::VariantType<Node> Element>
  constexpr int index_of(const Element& node) const {
    int i = mp::index_of(nodes_, node);
    assert(0 <= i && i < M);
    return i;
  }

  /// Access to the underlying data.
  constexpr decltype(auto) operator[](int i) const {
    assert(0 <= i && i < M);
    return nodes_[i];
  }

  /// Get the left child of a Binary node.
  constexpr decltype(auto) left(Binary auto& node) const {
    return nodes_[left_[index_of(node)]];
  }

  /// Get the right child of a Unary or Binary node.
  template <typename T> requires(Unary<T> || Binary<T>)
  constexpr decltype(auto) right(const T& node) const {
    return nodes_[index_of(node) - 1];
  }

  /// Rebind the index in a tree.
  constexpr Tree<M + 1> operator()(auto... is) const {
    return Tree<M + 1>(*this, Bind((is + ... + Index())));
  }

  /// Get the outer index of the tree.
  friend constexpr Index outer(const Tree& tree) {
    return std::visit([](auto&& node) {
      return outer(std::forward<decltype(node)>(node));
    }, tree.nodes_[M-1]);
  }
};

template <int A, int B>
Tree(const Tree<A>&, const Tree<B>&, Binary auto) -> Tree<A + B + 1>;

template <int A>
Tree(const Tree<A>&, Unary auto) -> Tree<A + 1>;

Tree(Leaf auto) -> Tree<1>;

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
  auto&&  l = bind(std::forward<A>(a));
  auto&&  r = bind(std::forward<B>(b));
  auto node = Sum(outer(l), outer(r));
  return Tree(std::forward<decltype(l)>(l), std::forward<decltype(r)>(r), node);
}

template <Expression A, Expression B>
constexpr auto operator*(A&& a, B&& b) {
  auto&&  l = bind(std::forward<A>(a));
  auto&&  r = bind(std::forward<B>(b));
  auto node = Product(outer(l), outer(r));
  return Tree(std::forward<decltype(l)>(l), std::forward<decltype(r)>(r), node);
}

template <Expression A, Expression B>
constexpr auto operator/(A&& a, B&& b) {
  auto&&  l = bind(std::forward<A>(a));
  auto&&  r = bind(std::forward<B>(b));
  auto node = Inverse(outer(l), outer(r));
  return Tree(std::forward<decltype(l)>(l), std::forward<decltype(r)>(r), node);
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
  auto&&  l = bind(std::forward<A>(a));
  auto node = Partial(outer(l), (is + ... + Index()));
  return Tree(std::forward<decltype(l)>(l), node);
}

constexpr Delta delta(Index a, Index b) {
  assert(a.size() == 1);
  assert(b.size() == 1);
  return Delta(a + b);
}

template <Expression A>
constexpr auto symmetrize(A&& a) {
  return Rational(1,2) * (a + a(reverse(outer(a))));
}
}
