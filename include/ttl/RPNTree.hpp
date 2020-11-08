#pragma once

#include "Index.hpp"
#include "Rational.hpp"
#include "Tensor.hpp"

namespace ttl {

enum Tag {
  SUM,
  DIFFERENCE,
  PRODUCT,
  INVERSE,
  BIND,
  PARTIAL,
  INDEX,
  DELTA,
  TENSOR,
  RATIONAL,
  DOUBLE
};

constexpr bool binary(Tag tag) {
  return tag < INDEX;
}

constexpr bool index(Tag tag) {
  return tag < TENSOR;
}

constexpr Index outer(Tag tag, Index a, Index b) {
  assert(binary(tag));
  switch (tag) {
   case SUM:
   case DIFFERENCE:
    assert(permutation(a, b));
    return a;
   case PRODUCT:
   case INVERSE:
    return a ^ b;
   case BIND:
   case PARTIAL:
    return exclusive(a + b);
   default:
    __builtin_unreachable();
  }
}

struct RPNNode {
  Tag tag;
  union {
    const Tensor* tensor;
    Index index;
    Rational q;
    double d;
  };
  int left{};

  constexpr RPNNode() {}
  constexpr RPNNode(const Tensor* tensor) : tag(TENSOR), tensor(tensor) {}
  constexpr RPNNode(Rational q) : tag(RATIONAL), q(q) {}
  constexpr RPNNode(double d) : tag(DOUBLE), d(d) {}
  constexpr RPNNode(Index i, Tag tag, int left = 0)
      : tag(tag)
      , index(i)
      , left(left)
  {}
};

template <int N = 1>
struct RPNTree final
{
  constexpr static std::true_type is_tree_tag = {};
  RPNNode nodes[N];

  constexpr RPNTree(Index index, Tag tag = INDEX) noexcept {
    assert(N == 1);
    nodes[0] = RPNNode(index, tag);
  }

  constexpr RPNTree(const Tensor* t) noexcept {
    assert(N == 1);
    nodes[0] = RPNNode(t);
  }

  constexpr RPNTree(Rational q) noexcept {
    assert(N == 1);
    nodes[0] = RPNNode(q);
  }

  constexpr RPNTree(double d) noexcept {
    assert(N == 1);
    nodes[0] = RPNNode(d);
  }

  template <int A, int B>
  constexpr RPNTree(Tag tag, const RPNTree<A>& a, const RPNTree<B>& b) noexcept
  {
    assert(A + B + 1 == N);
    assert(binary(tag));
    std::copy_n(a.nodes, A, nodes);
    std::copy_n(b.nodes, B, nodes + A);
    nodes[N - 1] = RPNNode(outer(tag, outer(a), outer(b)), tag, B + 1);
  }

  constexpr const RPNNode& operator[](int i) const { return nodes[i]; }
  constexpr       RPNNode& operator[](int i)       { return nodes[i]; }

  constexpr Tag     tag(int i) const { return nodes[i].tag; }
  constexpr Index index(int i) const {
    return (ttl::index(nodes[i].tag)) ? nodes[i].index : Index();
  }

  constexpr RPNTree operator()(std::same_as<Index> auto const&... is) const {
    RPNTree  copy = *this;
    Index  search = outer(copy);
    Index replace = (is + ... + Index());
    assert(search.size() == replace.size());

    for (RPNNode& node : copy.nodes) {
      if (ttl::index(node.tag)) {
        node.index.search_and_replace(search, replace);
      }
    }

    return copy;
  }

  constexpr friend int size(const RPNTree&) {
    return N;
  }

  constexpr friend Index outer(const RPNTree& tree) {
    return tree.index(N - 1);
  }
};

template <int A, int B>
RPNTree(Tag, RPNTree<A>, RPNTree<B>) -> RPNTree<A + B + 1>;

template <typename T>
concept is_expression =
 is_tree<T> ||
 std::same_as<T, Tensor> ||
 std::same_as<T, Index> ||
 std::same_as<T, Rational> ||
 std::signed_integral<T> ||
 std::same_as<T, double>;

constexpr auto bind(is_expression auto a) {
  return RPNTree(a);
}

constexpr auto bind(const Tensor& t) {
  return RPNTree(std::addressof(t));
}

constexpr auto bind(std::signed_integral auto t) {
  return bind(Rational(t));
}

constexpr auto bind(is_tree auto a) {
  return a;
}

constexpr auto operator+(is_expression auto const& a) {
  return bind(a);
}

constexpr auto operator+(is_expression auto const& a, is_expression auto const& b) {
  return RPNTree(SUM, bind(a), bind(b));
}

constexpr auto operator*(is_expression auto const& a, is_expression auto const& b) {
  return RPNTree(PRODUCT, bind(a), bind(b));
}

constexpr auto operator-(is_expression auto const& a, is_expression auto const& b) {
  return RPNTree(DIFFERENCE, bind(a), bind(b));
}

constexpr auto operator-(is_expression auto const& a) {
  return Rational(-1) * bind(a);
}

constexpr auto operator/(is_expression auto const& a, is_expression auto const& b) {
  return RPNTree(INVERSE, bind(a), bind(b));
}

constexpr auto D(is_expression auto const& a, const Index& i, std::same_as<Index> auto const&... is) {
  return RPNTree(PARTIAL, bind(a), RPNTree((i + ... + is)));
}

constexpr RPNTree<1> delta(const Index& a, const Index& b) {
  assert(a.size() == 1);
  assert(b.size() == 1);
  assert(a != b);
  return RPNTree(a + b, DELTA);
}

constexpr auto symmetrize(is_expression auto const& a) {
  RPNTree t = bind(a);
  return Rational(1,2) * (t + t(reverse(outer(t))));
}

constexpr auto
Tensor::operator()(std::same_as<Index> auto const&... is) const {
  Index i = (is + ... + Index());
  assert(i.size() == order_);
  return RPNTree(BIND, RPNTree(this), RPNTree(i));
}
}
