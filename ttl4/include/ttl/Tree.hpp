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

// placing double first gives this a useful default constructor
using Node = std::variant<double, Bind, Product, Sum, Inverse, Partial, Delta, TensorRef, Rational>;

template <int M>
class Tree
{
  int n_ = 0;
  Node data_[M];

  constexpr Tree(const Node* begin, int n) : n_(n)
  {
    assert(n_ <= M);
    std::copy_n(begin, n, data_);
  }

 public:
  constexpr Tree() = delete;
  constexpr Tree(const Tree&) = delete;
  constexpr Tree(Tree&&) = delete;

  /// Join two trees with a binary node.
  template <int A, int B>
  constexpr Tree(Binary auto node, const Tree<A>& a, const Tree<B>& b)
  {
    for (auto&& i : a) {
      data_[n_++] = i;
    }
    for (auto&& i : b) {
      data_[n_++] = i;
    }
    data_[n_++] = Node(node);
  }

  /// Wrap a tree in a unary node.
  template <int A>
  constexpr Tree(Unary auto node, const Tree<A>& a)
  {
    for (auto&& i : a) {
      data_[n_++] = i;
    }
    data_[n_++] = Node(node);
  }

  /// Construct a tree from a leaf node.
  constexpr Tree(Leaf auto node) : n_(1), data_ { node }
  {
  }

  constexpr int size() const {
    return n_;
  }

  constexpr int capacity() const {
    return M;
  }

  constexpr auto begin() const {
    return data_;
  }

  constexpr auto end() const {
    return data_ + n_;
  }

  /// Access to the underlying data.
  constexpr decltype(auto) operator[](int i) const {
    assert(0 <= i && i < n_);
    return data_[i];
  }

  /// Access to the top of the tree.
  constexpr decltype(auto) back() const {
    assert(n_ > 0);
    return data_[n_ - 1];
  }

  constexpr void push(const Node& node) {
    assert(n_ < M);
    data_[n_++] = node;
  }

  constexpr void push(Node&& node) {
    assert(n_ < M);
    data_[n_++] = node;
  }

  /// Rebind the index in a tree.
  constexpr Tree<M + 1> operator()(auto... is) const {
    return Tree<M + 1>(Bind((is + ... + Index())), *this);
  }

  template <typename Op>
  constexpr auto visit(Op&& op) const {
    using State = decltype(op(0, 1.0));
    // mp::vector<State> stack;
    mp::cvector<State, M> stack;
    for (int i = 0; const Node& n : data_) {
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

  constexpr int left() const {
    return visit(mp::overloaded {
        [](int i, Binary auto&&, int a, int b) {
          assert(b == i - 1);
          return a;
        },
        [](int i, Unary auto&&, int a) {
          assert(a == i - 1);
          return a;
        },
        [](int i, Leaf auto&&) {
          return i;
        }});
  }

  constexpr std::tuple<Node, Tree<M-1>, Tree<M-1>> split() const {
    // Find the left child of the root of the tree and split the tree, removing the
    // top node.
    int n = left();
    int m = size() - n;
    return std::tuple(data_[M - 1], Tree(data_, n), Tree(data_ + n, m));
  }
};

template <int A, int B>
Tree(Binary auto, Tree<A>, Tree<B>) -> Tree<A + B + 1>;

template <int A>
Tree(Unary auto, Tree<A>) -> Tree<A + 1>;

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
template <typename Op, int M>
constexpr auto visit(Op&& op, const Tree<M>& tree) {
  return tree.visit(std::forward<Op>(op));
}

/// We need to execute the tree in order to know what its outer index is
/// supposed to be.
template <int M>
constexpr Index outer(const Tree<M>& tree) {
  return tree.visit([&](int, const auto& node, auto&&... args) {
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
  return Tree(Bind((is + ... + Index())), Tree(std::cref(*this)));
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
  return Tree(Sum(), bind(std::forward<A>(a)), bind(std::forward<B>(b)));
}

template <Expression A, Expression B>
constexpr auto operator*(A&& a, B&& b) {
  return Tree(Product(), bind(std::forward<A>(a)), bind(std::forward<B>(b)));
}

template <Expression A, Expression B>
constexpr auto operator/(A&& a, B&& b) {
  return Tree(Inverse(), bind(std::forward<A>(a)), bind(std::forward<B>(b)));
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
  // if constexpr (is_tree_v<std::remove_cvref_t<A>(a)) {
  //   auto&& [b, left, right] = split(std::forward<A>(a));
  //   return std::visit([]<typename T>(const T& node) {
  //       if constexpr (std::same_as<Sum, std::remove_cvref_t<T>) {
  //         // M + M + 1
  //         return D(std::forward<decltype(left)>(left), is...) + D(std::forward<decltype(right)>(right), is...);
  //       }
  //       else {
  //         return
  //       }
  //   }, b);
  // }
  // else {
    return Tree(Partial((is + ... + Index())), bind(std::forward<A>(a)));
  // }
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
