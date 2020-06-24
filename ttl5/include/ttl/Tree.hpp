#pragma once

#include "Index.hpp"
#include "Nodes.hpp"
#include "Rational.hpp"
#include "Tensor.hpp"
#include "mp/cvector.hpp"
#include <concepts>
#include <functional>
#include <variant>

namespace ttl
{
struct Node
{
  union {
    std::monostate _;
    Sum sum;
    Product product;
    Inverse inverse;
    Bind bind;
    Partial partial;
    Delta delta;
    TensorRef tensor;
    Rational q;
    double d;
  };

  constexpr Node() : _() {}
  constexpr Node(const Sum& sum) : sum(sum) {}
  constexpr Node(const Product& product) : product(product) {}
  constexpr Node(const Inverse& inverse) : inverse(inverse) {}
  constexpr Node(const Bind& bind) : bind(bind) {}
  constexpr Node(const Partial& partial) : partial(partial) {}
  constexpr Node(const Delta& delta) : delta(delta) {}
  constexpr Node(TensorRef tensor) : tensor(tensor) {}
  constexpr Node(const Rational& q) : q(q) {}
  constexpr Node(const double& d) : d(d) {}

  template <typename Op, ValidTag Tag>
  constexpr auto visit(Op&& op, Tag tag) const {
    if constexpr (Tag::value == SUM) {
      return op(sum);
    }
    if constexpr (Tag::value == PRODUCT) {
      return op(product);
    }
    if constexpr (Tag::value == INVERSE) {
      return op(inverse);
    }
    if constexpr (Tag::value == BIND) {
      return op(bind);
    }
    if constexpr (Tag::value == PARTIAL) {
      return op(partial);
    }
    if constexpr (Tag::value == DELTA) {
      return op(delta);
    }
    if constexpr (Tag::value == TENSOR) {
      return op(tensor);
    }
    if constexpr (Tag::value == RATIONAL) {
      return op(q);
    }
    if constexpr (Tag::value == DOUBLE) {
      return op(d);
    }
  }
};

template <NodeType... Types>
class Tree
{
  template <NodeType...> friend class Tree;

  static constexpr int M = sizeof...(Types);
  Node data_[M];

 public:
  constexpr Tree() = delete;
  constexpr Tree(const Tree&) = delete;
  constexpr Tree(Tree&&) = delete;

  template <Binary T, NodeType... As, NodeType... Bs>
  constexpr Tree(const T& data, const Tree<As...>& a, const Tree<Bs...>& b) {
    int i = 0;
    for (const Node& data : a.data_) {
      data_[i++] = data;
    }
    for (const Node& data : b.data_) {
      data_[i++] = data;
    }
  }

  template <Unary T, NodeType... As>
  constexpr Tree(const T& data, const Tree<As...>& a) {
    int i = 0;
    for (const Node& data : a.data_) {
      data_[i++] = data;
    }
    data_[i++] = data;
    assert(i == M);
  }

  template <Leaf T>
  constexpr Tree(const T& data) {
    data_[0] = data;
  }

  template <IsIndex... Is>
  constexpr Tree<Types..., BIND> operator()(Is... is) const {
    return { Bind((is + ... + Index())), *this };
  }

  template <typename Op>
  constexpr auto visit(Op&& op) const {
    using State = decltype(op(0, 1.0));
    mp::cvector<State, M> stack;
    int i = 0;
    auto expand = [&](const auto& data) {
      if constexpr (Binary<decltype(data)>) {
        auto b = stack.pop();
        auto a = stack.pop();                   // could use back()
        stack.push(op(i, data, a, b));
      }
      else if constexpr(Unary<decltype(data)>) {
        auto a = stack.pop();                   // could use back()
        stack.push(op(i, data, a));
      }
      else {
        assert(Leaf<decltype(data)>);
        stack.push(op(i, data));
      }
    };
    ((data_[i].visit(expand, tag<Types>), ++i), ...);
    assert(i == M);
    assert(stack.size() == 1);
    return stack.pop();
  }
};

template <Binary T, NodeType... As, NodeType... Bs>
constexpr Tree<As..., Bs..., type_of<T>> make_tree(const T& data, const Tree<As...>& a, const Tree<Bs...>& b) {
  return { data, a, b };
}

template <Unary T, NodeType... As>
constexpr Tree<As..., type_of<T>> make_tree(const T& data, const Tree<As...>& a) {
  return { data, a };
}

template <Leaf T>
constexpr Tree<type_of<T>> make_tree(const T& data) {
  return { data };
}

template <typename>       constexpr inline bool is_tree_v = false;
template <NodeType... Ts> constexpr inline bool is_tree_v<Tree<Ts...>> = true;

template <typename T>
concept Expression =
 std::same_as<std::remove_cvref_t<T>, Tensor> ||
 std::same_as<std::remove_cvref_t<T>, Rational> ||
 std::same_as<std::remove_cvref_t<T>, double> ||
 std::signed_integral<std::remove_cvref_t<T>> ||
 is_tree_v<std::remove_cvref_t<T>>;

/// We need to visit a tree in order to know what its outer index is supposed to
/// be.
template <typename Tree> requires is_tree_v<Tree>
constexpr Index outer(const Tree& tree) {
  return tree.visit([&](int, const auto& node, auto&&... args) {
    return outer(node, std::forward<decltype(args)>(args)...);
  });
}

constexpr auto bind(const Tensor& t) {
  return make_tree(std::cref(t))();
}

constexpr auto bind(Rational q) {
  return make_tree(q);
}

constexpr auto bind(double d) {
  return make_tree(d);
}

template <std::signed_integral T>
constexpr auto bind(T t) {
  return bind(Rational(t));
}

template <typename A> requires is_tree_v<std::remove_cvref_t<A>>
constexpr decltype(auto) bind(A&& a) {
  return std::forward<A>(a);
}

constexpr Tree<TENSOR, BIND> Tensor::operator()(IsIndex auto... is) const {
  return make_tree(std::cref(*this))(is...);
}

template <Expression A>
constexpr auto operator+(A&& a) {
  return std::forward<A>(a);
}

template <Expression A, Expression B>
constexpr auto operator+(A&& a, B&& b) {
  return make_tree(Sum(), bind(std::forward<A>(a)), bind(std::forward<B>(b)));
}

template <Expression A, Expression B>
constexpr auto operator*(A&& a, B&& b) {
  return make_tree(Product(), bind(std::forward<A>(a)), bind(std::forward<B>(b)));
}

template <Expression A, Expression B>
constexpr auto operator/(A&& a, B&& b) {
  return make_tree(Inverse(), bind(std::forward<A>(a)), bind(std::forward<B>(b)));
}

template <Expression A>
constexpr auto D(A&& a, IsIndex auto... is) {
  return make_tree(Partial((is + ...)), bind(std::forward<A>(a)));
}

template <Expression A>
constexpr auto operator-(A&& a) {
  return Rational(-1) * bind(std::forward<A>(a));
}

template <Expression A, Expression B>
constexpr auto operator-(A&& a, B&& b) {
  return bind(std::forward<A>(a)) + (-std::forward<B>(b));
}

constexpr auto delta(Index a, Index b) {
  assert(a.size() == 1);
  assert(b.size() == 1);
  return make_tree(Delta(a + b));
}

template <Expression A>
constexpr auto symmetrize(A&& a) {
  return Rational(1,2) * (bind(a) + a(reverse(outer(a))));
}
}
