#pragma once

#include "cvector.hpp"
#include "index.hpp"
#include "rational.hpp"
#include "tensor.hpp"

#include <cassert>
#include <concepts>
#include <functional>
#include <string_view>
#include <variant>

#include <iostream>

namespace ttl
{
struct Delta {
  index a;

  friend constexpr index outer(Delta d) {
    return d.a;
  }
};

struct Inverse {
  friend constexpr index outer(Inverse, index a, index b) {
    return a ^ b;
  }
};

struct Product {
  friend constexpr index outer(Product, index a, index b) {
    return a ^ b;
  }
};

struct Sum {
  friend constexpr index outer(Sum, index a, index b) {
    assert(permutation(a, b));
    return a;
  }
};

struct Partial {
  friend constexpr index outer(Partial, index a, index b) {
    return a ^ b;
  }
};

struct Bind {
  friend constexpr index outer(Bind, index, index b) {
    return exclusive(b);
  }
};

constexpr index outer(const tensor&) {
  return {};
}

constexpr index outer(index a) {
  return a;
}

constexpr index outer(double) {
  return {};
}

constexpr index outer(rational) {
  return {};
}

template <typename T>
concept Internal =
 std::same_as<Inverse, std::remove_cvref_t<T>> ||
 std::same_as<Product, std::remove_cvref_t<T>> ||
 std::same_as<Sum,     std::remove_cvref_t<T>> ||
 std::same_as<Partial, std::remove_cvref_t<T>> ||
 std::same_as<Bind,    std::remove_cvref_t<T>>;

template <typename T>
concept Leaf =
 std::same_as<index,  std::remove_cvref_t<T>> ||
 std::same_as<double, std::remove_cvref_t<T>> ||
 std::same_as<Delta,  std::remove_cvref_t<T>> ||
 Rational<T> ||
 Tensor<T>;

using Node = std::variant<index, Inverse, Product, Sum, Partial, Bind, Delta,
                          std::reference_wrapper<const tensor>, rational, double>;

template <int M>
class Tree {
  Node nodes_[M];

 public:
  constexpr Tree() = default;
  constexpr Tree(const Tree&) = default;
  constexpr Tree(Tree&&) = default;

  constexpr Tree(const tensor& t, index i, Bind b)
      : nodes_{ std::ref(t), i, b }
  {
  }

  template <int A, std::size_t... As>
  constexpr Tree(Tree<A> a, index i, Partial p, std::index_sequence<As...>)
      : nodes_{ a[As]..., i, p }
  {
  }

  template <int A>
  constexpr Tree(Tree<A> a, index i, Partial p)
      : Tree(a, i, p, std::make_index_sequence<A>())
  {
  }

  template <int A, int B, typename T, std::size_t... As, std::size_t... Bs>
  constexpr Tree(Tree<A> a, Tree<B> b, T t, std::index_sequence<As...>, std::index_sequence<Bs...>)
      : nodes_{ a[As]..., b[Bs]..., t }
  {
  }

  template <int A, int B, typename T>
  constexpr Tree(Tree<A> a, Tree<B> b, T t)
      : Tree(a, b, t, std::make_index_sequence<A>(), std::make_index_sequence<B>())
  {
  }

  template <Leaf A, int B, typename T, std::size_t... Bs>
  constexpr Tree(A&& a, Tree<B> b, T t, std::index_sequence<Bs...>)
      : nodes_{ std::forward<A>(a), b[Bs]..., t }
  {
  }

  template <Leaf A, int B, typename T>
  constexpr Tree(A&& a, Tree<B> b, T t)
      : Tree(std::forward<A>(a), b, t, std::make_index_sequence<B>())
  {
  }

  template <int A, Leaf B, typename T, std::size_t... As>
  constexpr Tree(Tree<A> a, B&& b, T t, std::index_sequence<As...>)
      : nodes_{ a[As]..., std::forward<B>(b), t }
  {
  }

  template <int A, Leaf B, typename T>
  constexpr Tree(Tree<A> a, B&& b, T t)
      : Tree(a, std::forward<B>(b), t, std::make_index_sequence<A>())
  {
  }

  constexpr Tree<M+2> operator()(Index auto a, Index auto... bs) const {
    return { *this, (a + ... + bs), Bind{} };
  }

  constexpr decltype(auto) operator[](int i) const {
    assert(0 <= i && i < M);
    return nodes_[i];
  }

  constexpr decltype(auto) operator[](int i) {
    assert(0 <= i && i < M);
    return nodes_[i];
  }

  constexpr auto begin() const { return nodes_; }
  constexpr auto   end() const { return nodes_ + M; }

  constexpr static int size() { return M; }

  friend constexpr int size(Tree) { return M; }
};

Tree() -> Tree<0>;
Tree(const tensor&, index, Bind) -> Tree<3>;

template <int M>
Tree(Tree<M>, index, Partial) -> Tree<M+2>;

template <int M, int N, typename T>
Tree(Tree<M>, Tree<N>, T) -> Tree<M+N+1>;

template <Leaf A, int B, typename T>
Tree(A&&, Tree<B>, T) -> Tree<B+2>;

template <int A, Leaf B, typename T>
Tree(Tree<A>, B&&, T) -> Tree<A+2>;

template <typename> inline constexpr bool is_tree_v = false;
template <int M>    inline constexpr bool is_tree_v<Tree<M>> = true;

template <typename T>
concept Expression = is_tree_v<std::remove_cvref_t<T>> || Leaf<T> ||  std::signed_integral<std::remove_cvref_t<T>>;

/// We need to execute the tree in order to know what its outer index is
/// supposed to be. In the future it might make sense to store this as part of
/// each Node so it can just be looked up.
template <int M>
constexpr index outer(const Tree<M>& tree) {
  cvector<index, M> stack;
  for (int i = 0; const Node& n : tree) {
    std::visit([&](auto&& expr) {
      if constexpr (Internal<decltype(expr)>) {
        index b = stack.pop();
        index a = stack.pop();
        index c = outer(expr, a, b);
        stack.push(c);
      }
      else {
        index a = outer(expr);
        stack.push(a);
      }
    }, n);
  }
  assert(stack.size() == 1);
  return stack.pop();
}

constexpr Tree<3> tensor::operator()(Index auto... is) const {
  return Tree(*this, (is + ... + index()), Bind{});
}

constexpr Tree<3> autobind(const tensor& t) {
  return { t, index{}, Bind{} };
}

constexpr rational autobind(std::ptrdiff_t a) {
  return { a };
}

constexpr auto autobind(Leaf auto&& a) {
  return a;
}

template <int M>
constexpr auto autobind(const Tree<M>& a) {
  return a;
}

template <int M>
constexpr auto autobind(Tree<M>&& a) {
  return a;
}

constexpr auto operator+(Expression auto&& a) {
  return a;
}

template <Expression A, Expression B>
constexpr auto operator+(A&& a, B&& b) {
  return Tree(autobind(std::forward<A>(a)), autobind(std::forward<B>(b)), Sum{});
}

template <Expression A, Expression B>
constexpr auto operator*(A&& a, B&& b) {
  return Tree(autobind(std::forward<A>(a)), autobind(std::forward<B>(b)), Product{});
}

template <Expression A, Expression B>
constexpr auto operator/(A&& a, B&& b) {
  if constexpr (Rational<B>) {
    return std::forward<A>(a) * b.inverse();
  }
  else {
    return Tree(autobind(std::forward<A>(a)), autobind(std::forward<B>(b)), Inverse{});
  }
}

template <Expression A>
constexpr auto operator-(A&& a) {
  return autobind(std::forward<A>(a)) * rational{-1};
}

template <Expression A, Expression B>
constexpr auto operator-(A&& a, B&& b) {
  return autobind(std::forward<A>(a)) + autobind(-std::forward<B>(b));
}

template <Expression A>
constexpr auto D(A&& a, Index auto... is) {
  return Tree(autobind(std::forward<A>(a)), (is + ...), Partial{});
}

constexpr Delta delta(index i, index j) {
  assert(i.size() == 1);
  assert(j.size() == 1);
  return { i + j };
}

constexpr auto symmetrize(Expression auto&& a) {
  return rational(1,2) * (a + a(reverse(outer(a))));
}
}
