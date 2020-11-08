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
  Tag tag = {};
  union {
    struct {} monostate = {};
    const Tensor* tensor;
    Index index;
    Rational q;
    double d;
  };
  int left = {};

  constexpr RPNNode() {}
};

template <int N = 1>
struct RPNTree final
{
  constexpr static std::true_type is_tree_tag = {};
  RPNNode nodes[N];

  constexpr RPNTree(Index index, Tag tag = INDEX) noexcept {
    assert(N == 1);
    nodes[0].tag = tag;
    nodes[0].index = index;
  }

  constexpr RPNTree(const Tensor* t) noexcept {
    assert(N == 1);
    nodes[0].tag = TENSOR;
    nodes[0].tensor = t;
  }

  constexpr RPNTree(Rational q) noexcept {
    assert(N == 1);
    nodes[0].tag = RATIONAL;
    nodes[0].q = q;
  }

  constexpr RPNTree(double d) noexcept {
    assert(N == 1);
    nodes[0].tag = DOUBLE;
    nodes[0].d = d;
  }

  template <int A, int B>
  constexpr RPNTree(Tag tag, const RPNTree<A>& a, const RPNTree<B>& b) noexcept
  {
    assert(A + B + 1 == N);
    assert(binary(tag));
    int i = 0;
    for (int j = 0; j < A; ++j, ++i) nodes[i] = a[j];
    for (int j = 0; j < B; ++j, ++i) nodes[i] = b[j];
    nodes[i].tag = tag;
    nodes[i].index = outer(tag, outer(a), outer(b));
    nodes[i].left = B + 1;
  }

  constexpr const RPNNode& operator[](int i) const { return nodes[i]; }
  constexpr       RPNNode& operator[](int i)       { return nodes[i]; }

  constexpr Tag     tag(int i) const { return nodes[i].tag; }
  constexpr Index index(int i) const {
    return (ttl::index(nodes[i].tag)) ? nodes[i].index : Index();
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

constexpr auto symmetrize(is_expression auto a) {
  RPNTree t = bind(a);
  Index i = t.outer();
  return Rational(1,2) * (t + t(reverse(i)));
}

constexpr auto
Tensor::operator()(std::same_as<Index> auto const&... is) const {
  Index i = (is + ... + Index());
  assert(i.size() == order_);
  return RPNTree(BIND, RPNTree(this), RPNTree(i));
}
}
