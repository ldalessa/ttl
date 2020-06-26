#pragma once

#include "Index.hpp"
#include "Nodes.hpp"
#include "Rational.hpp"
#include "mp/apply.hpp"
#include "mp/cvector.hpp"
#include <concepts>
#include <functional>
#include <tuple>
#include <variant>

namespace ttl
{
enum NodeType : char {
  SUM,
  DIFFERENCE,
  PRODUCT,
  INVERSE,
  BIND,
  PARTIAL,
  DELTA,
  TENSOR,
  RATIONAL,
  DOUBLE,
  INVALID
};

constexpr int is_binary_type(NodeType type) {
  return type < BIND;
}

constexpr int is_unary_type(NodeType type) {
  return BIND <= type && type < DELTA;
}

constexpr int is_leaf_type(NodeType type) {
  return DELTA <= type && type < INVALID;
}

template <typename> constexpr inline NodeType type_of = INVALID;

template <> constexpr inline NodeType type_of<Sum>        = SUM;
template <> constexpr inline NodeType type_of<Difference> = DIFFERENCE;
template <> constexpr inline NodeType type_of<Product>    = PRODUCT;
template <> constexpr inline NodeType type_of<Inverse>    = INVERSE;
template <> constexpr inline NodeType type_of<Bind>       = BIND;
template <> constexpr inline NodeType type_of<Partial>    = PARTIAL;
template <> constexpr inline NodeType type_of<Delta>      = DELTA;
template <> constexpr inline NodeType type_of<TensorRef>  = TENSOR;
template <> constexpr inline NodeType type_of<Rational>   = RATIONAL;
template <> constexpr inline NodeType type_of<double>     = DOUBLE;

struct Node
{
  union {
    std::monostate _;
    Sum sum;
    Difference difference;
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
  constexpr Node(const Difference& difference) : difference(difference) {}
  constexpr Node(const Product& product) : product(product) {}
  constexpr Node(const Inverse& inverse) : inverse(inverse) {}
  constexpr Node(const Bind& bind) : bind(bind) {}
  constexpr Node(const Partial& partial) : partial(partial) {}
  constexpr Node(const Delta& delta) : delta(delta) {}
  constexpr Node(TensorRef tensor) : tensor(tensor) {}
  constexpr Node(const Rational& q) : q(q) {}
  constexpr Node(const double& d) : d(d) {}

  /// Standard visit functionality just calls op with the cast union.
  template <typename Op>
  constexpr auto visit(Op&& op, NodeType Type) const {
    switch (Type) {
     case SUM:        return op(sum);
     case DIFFERENCE: return op(difference);
     case PRODUCT:    return op(product);
     case INVERSE:    return op(inverse);
     case BIND:       return op(bind);
     case PARTIAL:    return op(partial);
     case DELTA:      return op(delta);
     case TENSOR:     return op(tensor);
     case RATIONAL:   return op(q);
     case DOUBLE:     return op(d);
     case INVALID:
     default:
      assert(false);
    }
    __builtin_unreachable();
  }

  template <typename Op>
  constexpr auto visit(Op&& op, NodeType Type) {
    switch (Type) {
     case SUM:        return op(sum);
     case DIFFERENCE: return op(difference);
     case PRODUCT:    return op(product);
     case INVERSE:    return op(inverse);
     case BIND:       return op(bind);
     case PARTIAL:    return op(partial);
     case DELTA:      return op(delta);
     case TENSOR:     return op(tensor);
     case RATIONAL:   return op(q);
     case DOUBLE:     return op(d);
     case INVALID:
     default:
      assert(false);
    }
    __builtin_unreachable();
  }

  /// Fancy visit op allows different return types based on Type.
  template <typename Op, NodeType Type>
  constexpr auto visit(Op&& op, std::integral_constant<NodeType, Type>) const {
    if constexpr (Type == SUM) {
      return op(sum);
    }
    if constexpr (Type == DIFFERENCE) {
      return op(difference);
    }
    if constexpr (Type == PRODUCT) {
      return op(product);
    }
    if constexpr (Type == INVERSE) {
      return op(inverse);
    }
    if constexpr (Type == BIND) {
      return op(bind);
    }
    if constexpr (Type == PARTIAL) {
      return op(partial);
    }
    if constexpr (Type == DELTA) {
      return op(delta);
    }
    if constexpr (Type == TENSOR) {
      return op(tensor);
    }
    if constexpr (Type == RATIONAL) {
      return op(q);
    }
    if constexpr (Type == DOUBLE) {
      return op(d);
    }
    __builtin_unreachable();
  }

  template <typename Op, NodeType Type>
  constexpr auto visit(Op&& op, std::integral_constant<NodeType, Type>) {
    if constexpr (Type == SUM) {
      return op(sum);
    }
    if constexpr (Type == DIFFERENCE) {
      return op(difference);
    }
    if constexpr (Type == PRODUCT) {
      return op(product);
    }
    if constexpr (Type == INVERSE) {
      return op(inverse);
    }
    if constexpr (Type == BIND) {
      return op(bind);
    }
    if constexpr (Type == PARTIAL) {
      return op(partial);
    }
    if constexpr (Type == DELTA) {
      return op(delta);
    }
    if constexpr (Type == TENSOR) {
      return op(tensor);
    }
    if constexpr (Type == RATIONAL) {
      return op(q);
    }
    if constexpr (Type == DOUBLE) {
      return op(d);
    }
    __builtin_unreachable();
  }
};

template <NodeType... Types>
class Tree
{
  template <NodeType...> friend class Tree;

  constexpr static int M = sizeof...(Types);
  constexpr static NodeType types_[M] = {Types...};
  Node data_[M];

  /// Count the number of nodes in a subtree.
  static constexpr int count(int i) {
    if (is_leaf_type(types_[i])) {
      return 1;
    }

    if (is_unary_type(types_[i])) {
      return count(i - 1) + 1;
    }

    // the count the number of nodes in the right subtree tells us where the
    // left subtree starts
    int right = count(i - 1);
    assert(0 <= i - right - 1);
    return count(i - right - 1) + right + 1;
  }

  constexpr void rebind_topdown(Index outer, int i, const Index is[M], const int left[M], const int right[M]) {
    using ttl::rebind;
    data_[i].visit([&](auto& node) {
      if constexpr (Binary<decltype(node)>) {
        auto&& [a, b] = rebind(node, outer, is[left[i]], is[right[i]]);
        rebind_topdown(b, right[i], is, left, right); // rebind right subtree
        rebind_topdown(a, left[i],  is, left, right); // rebind left subtree
      }
      else if constexpr (Unary<decltype(node)>) {
        auto&& [a] = rebind(node, outer, is[left[i]]);
        rebind_topdown(a, left[i], is, left, right); // rebind child subtree
      }
      else {
        assert(Leaf<decltype(node)>);
        rebind(node, outer);
      }
    }, types_[i]);
  }

  /// Rebind the indices in the tree.
  constexpr auto rebind(Index a) && {
    Index   is[M];
    int   left[M];
    int  right[M];

    // Perform a bottom-up pass to build the set of outer indices, and to record
    // the geometry for the top-down pass.
    visit([&](int i, const auto& node, auto... children) {
      assert(((children < i) && ...));
      is[i] = outer(node, is[children]...);
      if constexpr (sizeof...(children) == 2) {
        [&](int a, int b) {
          left[i] = a;
          right[i] = b;
        }(children...);
      }
      else if constexpr (sizeof...(children) == 1) {
        [&](int a) {
          left[i] = a;
          right[i] = -1;
        }(children...);
      }
      else {
        left[i] = -1;
        right[i] = -1;
      }
      return i;
    });

    rebind_topdown(a, M-1, is, left, right);

    return std::move(*this);
  }

  constexpr explicit Tree(const Node *begin) {
    std::copy_n(begin, M, data_);
  }

 public:
  constexpr static NodeType type() {
    return types_[M-1];
  }

  constexpr static bool is_constant() {
    return ((Types != TENSOR) && ...);
  }

  constexpr Tree() = delete;

  constexpr Tree(const Tree& rhs) {
    for (int i = 0; i < M; ++i) {
      rhs.data_[i].visit([&](const auto& node) {
        data_[i] = node;
      }, rhs.types_[i]);
    }
  }

  constexpr Tree(Tree&& rhs) {
    for (int i = 0; i < M; ++i) {
      rhs.data_[i].visit([&](const auto& node) {
        data_[i] = node;
      }, rhs.types_[i]);
    }
  }

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
  constexpr auto operator()(Is... is) && {
    return std::move(*this).rebind((is + ...));
  }

  template <IsIndex... Is>
  constexpr auto operator()(Is... is) const & {
    return Tree(*this)(is...);
  }

  template <typename Op>
  constexpr auto visit_root(Op&& op) const {
    return data_[M-1].visit(std::forward<Op>(op), std::integral_constant<NodeType, types_[M-1]>());
  }

  constexpr Tree& extend_partial(Index i) & {
    assert(type() == PARTIAL);
    data_[M-1].partial.extend(i);
    return *this;
  }

  constexpr Tree&& extend_partial(Index i) && {
    assert(type() == PARTIAL);
    data_[M-1].partial.extend(i);
    return std::move(*this);
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
    ((data_[i].visit(expand, Types), ++i), ...);
    assert(i == M);
    assert(stack.size() == 1);
    return stack.pop();
  }

  constexpr auto split() const {
    assert(2 <= M);
    constexpr int R = count(M - 2);      // number of nodes in the right subtree
    constexpr int L = M - R - 1;         // number of nodes in the left subtree

    // create the left subtree by unpacking the first L nodes
    auto left = mp::apply([&](auto... is) {
      return ttl::Tree<types_[is()]...>(data_);
    }, std::make_index_sequence<L>());

    // create the right subtree by unpacking the next R nodes
    auto right = mp::apply([&](auto... is) {
      return ttl::Tree<types_[L + is()]...>(data_ + L);
    }, std::make_index_sequence<R>());

    // return the tuple
    return std::tuple(std::move(left), std::move(right));
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

template <NodeType... Ts> constexpr inline NodeType type_of<Tree<Ts...>> = Tree<Ts...>::type();

template <typename>       constexpr inline bool is_tree_v = false;
template <NodeType... Ts> constexpr inline bool is_tree_v<Tree<Ts...>> = true;

template <typename>       constexpr inline bool is_tree_constant_v = false;
template <NodeType... Ts> constexpr inline bool is_tree_constant_v<Tree<Ts...>> = ((Ts != TENSOR) && ...);

template <typename>       constexpr inline bool is_constant_v = false;
template <NodeType... Ts> constexpr inline bool is_constant_v<Tree<Ts...>> = Tree<Ts...>::is_constant();

/// We need to visit a tree in order to know what its outer index is supposed to
/// be.
template <typename Tree> requires is_tree_v<Tree>
constexpr Index outer(const Tree& tree) {
  return tree.visit([&](int, const auto& node, auto&&... args) {
    return outer(node, std::forward<decltype(args)>(args)...);
  });
}

template <typename T>
concept Expression =
 std::same_as<std::remove_cvref_t<T>, Tensor> ||
 std::same_as<std::remove_cvref_t<T>, Rational> ||
 std::same_as<std::remove_cvref_t<T>, double> ||
 std::signed_integral<std::remove_cvref_t<T>> ||
 is_tree_v<std::remove_cvref_t<T>>;

constexpr auto bind(const Tensor& t) {
  return make_tree(std::cref(t));
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
constexpr auto operator-(A&& a) {
  return Rational(-1) * bind(std::forward<A>(a));
}

template <Expression A, Expression B>
constexpr auto operator-(A&& a, B&& b) {
  return make_tree(Difference(), bind(std::forward<A>(a)), bind(std::forward<B>(b)));
}

template <Expression A>
constexpr auto D(A&& a, IsIndex auto... is) {
  assert(sizeof...(is) != 0);

  // if constexpr (is_constant_v<std::remove_cvref_t<A>>) {
  //   return make_tree(Rational(0));
  // }
  // else
    if constexpr (type_of<std::remove_cvref_t<A>> == SUM) {
    auto&& [l, r] = a.split();
    return D(std::forward<decltype(l)>(l), is...) + D(std::forward<decltype(r)>(r), is...);
  }
  else if constexpr (type_of<std::remove_cvref_t<A>> == DIFFERENCE) {
    auto&& [l, r] = a.split();
    return D(std::forward<decltype(l)>(l), is...) - D(std::forward<decltype(r)>(r), is...);
  }
  else if constexpr (type_of<std::remove_cvref_t<A>> == PRODUCT) {
    auto&& [l, r] = a.split();
    if constexpr (is_constant_v<std::remove_cvref_t<decltype(r)>>) {
      return D(std::forward<decltype(l)>(l), is...) * std::forward<decltype(r)>(r);
    }
    else if constexpr (is_constant_v<std::remove_cvref_t<decltype(l)>>) {
      return std::forward<decltype(l)>(l) * D(std::forward<decltype(r)>(r), is...);
    }
    else {
      return D(l, is...) * r + l * D(r, is...);
    }
  }
  // else if constexpr (type_of<std::remove_cvref_t<A>> == PARTIAL) {
  //   return std::forward<A>(a).extend_partial((is + ...));
  // }
  else if constexpr (type_of<std::remove_cvref_t<A>> == INVERSE) {
    auto&& [l, r] = a.split();
    if constexpr (is_constant_v<std::remove_cvref_t<decltype(r)>>) {
      return D(std::forward<decltype(l)>(l), is...) / std::forward<decltype(r)>(r);
    }
    else if constexpr (is_constant_v<std::remove_cvref_t<decltype(l)>>) {
      return - l * D(r, is...) / (r * r);
    }
    else {
      return (D(l, is...) * r - l * D(r, is...)) / (r * r);
    }
  }
  else {
    // assert(type_of<std::remove_cvref_t<A>> == BIND ||
    //        type_of<std::remove_cvref_t<A>> == TENSOR);
    return make_tree(Partial((is + ...)), bind(std::forward<A>(a)));
  }
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
