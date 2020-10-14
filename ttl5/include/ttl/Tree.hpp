#pragma once

#include "Index.hpp"
#include "Nodes.hpp"
#include "Rational.hpp"
#include <fmt/core.h>
#include <cassert>
#include <concepts>
#include <functional>
#include <tuple>
#include <type_traits>
#include <variant>

namespace ttl
{

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

enum NodeTag : char {
  SUM,
  DIFFERENCE,
  PRODUCT,
  INVERSE,
  BIND,
  PARTIAL,
  DELTA,
  ZERO,
  ONE,
  TENSOR,
  RATIONAL,
  DOUBLE,
  INVALID
};

constexpr int is_binary(NodeTag type) {
  return type < BIND;
}

constexpr int is_unary(NodeTag type) {
  return BIND <= type && type < DELTA;
}

constexpr int is_leaf(NodeTag type) {
  return DELTA <= type && type < INVALID;
}

template <typename> constexpr inline NodeTag tag             = INVALID;
template <>         constexpr inline NodeTag tag<Sum>        = SUM;
template <>         constexpr inline NodeTag tag<Difference> = DIFFERENCE;
template <>         constexpr inline NodeTag tag<Product>    = PRODUCT;
template <>         constexpr inline NodeTag tag<Inverse>    = INVERSE;
template <>         constexpr inline NodeTag tag<Bind>       = BIND;
template <>         constexpr inline NodeTag tag<Partial>    = PARTIAL;
template <>         constexpr inline NodeTag tag<Delta>      = DELTA;
template <>         constexpr inline NodeTag tag<Zero>       = ZERO;
template <>         constexpr inline NodeTag tag<One>        = ONE;
template <>         constexpr inline NodeTag tag<TensorRef>  = TENSOR;
template <>         constexpr inline NodeTag tag<Rational>   = RATIONAL;
template <>         constexpr inline NodeTag tag<double>     = DOUBLE;

constexpr bool is_type(auto&& node, NodeTag type) {
  return tag<std::remove_cvref_t<decltype(node)>> == type;
}

union Node
{
  struct {}  _;                             // provides a default active variant
  Sum        sum;
  Difference difference;
  Product    product;
  Inverse    inverse;
  Bind       bind;
  Partial    partial;
  Delta      delta;
  Zero       zero;
  One        one;
  TensorRef  tensor;
  Rational   q;
  double     d;

  constexpr  Node()                      : _()                    {}
  constexpr  Node(Sum sum)               : sum(sum)               {}
  constexpr  Node(Difference difference) : difference(difference) {}
  constexpr  Node(Product product)       : product(product)       {}
  constexpr  Node(Inverse inverse)       : inverse(inverse)       {}
  constexpr  Node(Bind bind)             : bind(bind)             {}
  constexpr  Node(Partial partial)       : partial(partial)       {}
  constexpr  Node(Delta delta)           : delta(delta)           {}
  constexpr  Node(Zero zero)             : zero(zero)             {}
  constexpr  Node(One one)               : one(one)               {}
  constexpr  Node(TensorRef tensor)      : tensor(tensor)         {}
  constexpr  Node(Rational q)            : q(q)                   {}
  constexpr  Node(double d)              : d(d)                   {}
  constexpr ~Node() = default;

  /// Standard visit functionality just calls op with the cast union.
  template <typename Op>
  constexpr auto visit(Op&& op, NodeTag Type) const {
    switch (Type) {
     case SUM:        return op(sum);
     case DIFFERENCE: return op(difference);
     case PRODUCT:    return op(product);
     case INVERSE:    return op(inverse);
     case BIND:       return op(bind);
     case PARTIAL:    return op(partial);
     case DELTA:      return op(delta);
     case ZERO:       return op(zero);
     case ONE:        return op(sum);
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
  constexpr auto visit(Op&& op, NodeTag Type) {
    switch (Type) {
     case SUM:        return op(sum);
     case DIFFERENCE: return op(difference);
     case PRODUCT:    return op(product);
     case INVERSE:    return op(inverse);
     case BIND:       return op(bind);
     case PARTIAL:    return op(partial);
     case DELTA:      return op(delta);
     case ZERO:       return op(zero);
     case ONE:        return op(sum);
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
  template <NodeTag Type, typename Op>
  constexpr auto visit(Op&& op, std::integral_constant<NodeTag, Type> = {}) const {
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
    if constexpr (Type == ZERO) {
      return op(zero);
    }
    if constexpr (Type == ONE) {
      return op(one);
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

  template <NodeTag Type, typename Op>
  constexpr auto visit(Op&& op, std::integral_constant<NodeTag, Type> = {}) {
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
    if constexpr (Type == ZERO) {
      return op(zero);
    }
    if constexpr (Type == ONE) {
      return op(one);
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

template <NodeTag... Tags>
class Tree
{
  template <NodeTag...> friend class Tree;

  constexpr static int M = sizeof...(Tags);
  constexpr static NodeTag tags_[M] = {Tags...};
  Node data_[M];

  /// Count the number of nodes in a subtree.
  static constexpr int count(int i) {
    if (is_leaf(tags_[i])) {
      return 1;
    }

    if (is_unary(tags_[i])) {
      return count(i - 1) + 1;
    }

    // the count of the number of nodes in the right subtree tells us where the
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
    }, tags_[i]);
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
  constexpr static NodeTag type() {
    return tags_[M-1];
  }

  constexpr static int count(NodeTag type) {
    return ((type == Tags) + ...);
  }

  constexpr static bool is_constant() {
    return count(TENSOR) == 0;
  }

  constexpr Tree() = default;
  constexpr Tree(const Tree& rhs) = default;
  constexpr Tree(Tree&& rhs) = default;

  /// Create a tree from two subtrees and a binary node.
  template <Binary T, NodeTag... As, NodeTag... Bs>
  constexpr Tree(const T& data, const Tree<As...>& a, const Tree<Bs...>& b) {
    int i = 0;

    for (const Node& data : a.data_) {
      data_[i++] = data;
    }
    for (const Node& data : b.data_) {
      data_[i++] = data;
    }
    data_[i++] = data;
    assert(i == M);
  }

  template <Unary T, NodeTag... As>
  constexpr Tree(const T& data, const Tree<As...>& a) {
    int i = 0;
    for (const Node& data : a.data_) {
      data_[i++] = data;
    }
    data_[i++] = data;
    assert(i == M);
  }

  template <Leaf T>
  constexpr Tree(const T& data) : data_ { data } {
  }

  constexpr auto begin() const { return std::begin(data_); }
  constexpr auto   end() const { return   std::end(data_); }

  template <typename... Ops>
  constexpr void for_each(Ops&&... ops) const {
    overloaded op = {
      [](...) {},
      std::forward<Ops>(ops)...
    };

    for (int i = 0; i < M; ++i) {
      data_[i].visit(op, tags_[i]);
    }
  }

  constexpr auto operator()(std::same_as<Index> auto... is) && {
    return std::move(*this).rebind((is + ... + Index()));
  }

  constexpr auto operator()(std::same_as<Index> auto... is) const & {
    return Tree(*this)(is...);
  }

  template <typename Op>
  constexpr auto visit_root(Op&& op) const {
    return data_[M-1].visit<tags_[M-1]>(std::forward<Op>(op));
  }

  constexpr Tree extend_partial(Index i) const {
    assert(M == 2 || M == 3);
    assert(tags_[0] == TENSOR);
    assert(tags_[1] == PARTIAL || tags_[1] == BIND);
    assert(tags_[M-1] == PARTIAL);
    Tree copy(*this);
    copy.data_[M-1].partial.extend(i);
    return copy;
  }

  /// Visit nodes in the tree in a bottom-up order.
  template <typename Op>
  constexpr auto visit(Op&& op) const {
    using State = decltype(op(0, 1.0));
    ce::cvector<State, M> stack;
    int i = 0;
    auto expand = [&](const auto& data) {
      if constexpr (Binary<decltype(data)>) {
        auto b = stack.pop_back();
        auto a = stack.pop_back();
        stack.push_back(op(i, data, a, b));
      }
      else if constexpr(Unary<decltype(data)>) {
        auto a = stack.pop_back();
        stack.push_back(op(i, data, a));
      }
      else {
        assert(Leaf<decltype(data)>);
        stack.push_back(op(i, data));
      }
    };
    ((data_[i].visit(expand, Tags), ++i), ...);
    assert(i == M);
    assert(size(stack) == 1);
    return stack.pop_back();
  }

  constexpr auto split() const {
    static_assert(is_binary(tags_[M-1]));

    constexpr int R = count(M - 2);      // number of nodes in the right subtree
    constexpr int L = M - R - 1;         // number of nodes in the left subtree

    // create the left subtree by unpacking the first L nodes
    auto left = [&]<auto... is>(std::index_sequence<is...>) {
      return ttl::Tree<tags_[is]...>(data_);
    }(std::make_index_sequence<L>());

    // create the right subtree by unpacking the next R nodes
    auto right = [&]<auto... is>(std::index_sequence<is...>) {
      return ttl::Tree<tags_[L + is]...>(data_ + L);
    }(std::make_index_sequence<R>());

    // return the tuple
    return std::tuple(std::move(left), std::move(right));
  }

  constexpr Index get_outer() const {
    return visit([&](int, const auto& node, auto&&... args) {
      return outer(node, std::forward<decltype(args)>(args)...);
    });
  }

  constexpr friend Index outer(const Tree& tree) {
    return tree.get_outer();
  }

  std::string get_name() const {
    return visit([&](int, const auto& node, auto&&... args) -> std::string {
      int i = 0;
      std::string out = "(";
      out.append(name(node));
      ((out.append((i++ < sizeof...(args)) ? ", " : ""), out.append(args)), ...);
      out.append(")");
      return out;
    });
  }

  friend std::string name(const Tree& tree) {
    return tree.get_name();
  }
};

template <Binary T, NodeTag... As, NodeTag... Bs>
constexpr Tree<As..., Bs..., tag<T>> make_tree(const T& data, const Tree<As...>& a, const Tree<Bs...>& b) {
  return { data, a, b };
}

template <Unary T, NodeTag... As>
constexpr Tree<As..., tag<T>> make_tree(const T& data, const Tree<As...>& a) {
  return { data, a };
}

template <Leaf T>
constexpr Tree<tag<T>> make_tree(const T& data) {
  return { data };
}

template <NodeTag... Ts> constexpr inline NodeTag tag<Tree<Ts...>> = Tree<Ts...>::type();

template <typename>      constexpr inline bool is_tree_v = false;
template <NodeTag... Ts> constexpr inline bool is_tree_v<Tree<Ts...>> = true;

template <typename>      constexpr inline bool is_tree_constant_v = false;
template <NodeTag... Ts> constexpr inline bool is_tree_constant_v<Tree<Ts...>> = ((Ts != TENSOR) && ...);

template <typename>      constexpr inline bool is_constant_v = false;
template <NodeTag... Ts> constexpr inline bool is_constant_v<Tree<Ts...>> = Tree<Ts...>::is_constant();

template <typename T>
concept Expression =
 std::same_as<T, Tensor> ||
 std::same_as<T, Rational> ||
 std::same_as<T, double> ||
 std::signed_integral<T> ||
 is_tree_v<T>;

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
constexpr auto operator+(const A& a) {
  return a;
}

template <Expression A, Expression B>
constexpr auto operator+(const A& a, const B& b) {
  if constexpr (tag<B> == ZERO) {
    return a;
  }
  else if constexpr (tag<A> == ZERO) {
    return b;
  }
  else {
    assert(permutation(outer(a), outer(b)));
    return make_tree(Sum(), bind(a), bind(b));
  }
}

template <Expression A, Expression B>
constexpr auto operator*(const A& a, const B& b) {
  if constexpr (tag<B> == ONE) {
    return a;
  }
  else if constexpr (tag<A> == ONE) {
    return b;
  }
  else if constexpr (tag<B> == ZERO) {
    return b;
  }
  else if constexpr (tag<A> == ZERO) {
    return a;
  }
  else {
    return make_tree(Product(), bind(a), bind(b));
  }
}

template <Expression A, Expression B>
constexpr auto operator/(const A& a, const B& b) {
  static_assert(tag<B> != ZERO);
  return make_tree(Inverse(), bind(a), bind(b));

  // if constexpr (tag<B> == ONE) {
  //   return a;
  // }
  // else if constexpr (tag<A> == ZERO) {
  //   return a;
  // }
  // else if constexpr (tag<B> == DOUBLE) {
  //   return a * (1/b);
  // }
  // else if constexpr (tag<B> == RATIONAL) {
  //   return a * b.inverse();
  // }
  // // else detect if a == b ... is this even possible?
  // else {
  //   return make_tree(Inverse(), bind(a), bind(b));
  // }
}

template <Expression A>
constexpr auto operator-(const A& a) {
  return Rational(-1) * a;
}

template <Expression A, Expression B>
constexpr auto operator-(const A& a, const B& b) {
  if constexpr (tag<B> == ZERO) {
    return a;
  }
  else if constexpr (tag<A> == ZERO) {
    return -b;
  }
  else {
    assert(permutation(outer(a), outer(b)));
    return make_tree(Difference(), bind(a), bind(b));
  }
}

template <Expression A>
constexpr auto D(const A& a, std::same_as<Index> auto... is) {
  assert(sizeof...(is) != 0);
  // return make_tree(Partial((is + ...)), bind(a));

  if constexpr (is_constant_v<A>) {
    return make_tree(Zero());
  }
  else if constexpr (tag<A> == SUM) {
    auto [l, r] = a.split();
    return D(l, is...) + D(r, is...);
  }
  else if constexpr (tag<A> == DIFFERENCE) {
    auto [l, r] = a.split();
    return D(l, is...) - D(r, is...);
  }
  else if constexpr (tag<A> == PRODUCT) {
    auto [l, r] = a.split();
    if constexpr (is_constant_v<decltype(r)>) {
      return D(l, is...) * r;
    }
    else if constexpr (is_constant_v<decltype(l)>) {
      return l * D(r, is...);
    }
    else {
      Tree  left = D(l, is...) * r;
      Tree right = l * D(r, is...);
      return left + right;
    }
  }
  else if constexpr (tag<A> == PARTIAL) {
    return a.extend_partial((is + ...));
  }
  else if constexpr (tag<A> == INVERSE) {
    auto [l, r] = a.split();
    if constexpr (is_constant_v<decltype(r)>) {
      return D(l, is...) / r;
    }
    else if constexpr (is_constant_v<decltype(l)>) {
      return - l * D(r, is...) / (r * r);
    }
    else {
      return (D(l, is...) * r - l * D(r, is...)) / (r * r);
    }
  }
  else {
    assert(tag<A> == BIND || tag<A> == TENSOR || (std::is_same_v<A, Tensor> == true));
    return make_tree(Partial((is + ...)), bind(a));
  }
}

constexpr auto delta(Index a, Index b) {
  assert(size(a) == 1);
  assert(size(b) == 1);
  return make_tree(Delta(a + b));
}

template <Expression A>
constexpr auto symmetrize(A&& a) {
  Index i = outer(a);
  std::ranges::reverse(i);
  return Rational(1,2) * (bind(a) + a(i));
}
}
