#pragma once

#include <concepts>
#include <type_traits>
#include <tuple>
#include <utility>

namespace ttl {
class Tensor;
using TensorRef = std::reference_wrapper<const Tensor>;

struct Sum {
  constexpr Index outer(Index a, Index b) const {
    assert(permutation(a, b));
    return a;
  }

  constexpr friend std::string_view name(const Sum&) {
    return "+";
  }

  friend std::ostream& operator<<(std::ostream& os, const Sum& sum) {
    return os << name(sum);
  }
};

struct Difference {
  constexpr Index outer(Index a, Index b) const {
    assert(permutation(a, b));
    return a;
  }

  constexpr friend std::string_view name(const Difference&) {
    return "-";
  }

  friend std::ostream& operator<<(std::ostream& os, const Difference& diff) {
    return os << name(diff);
  }
};

struct Product {
  constexpr Index outer(Index a, Index b) const {
    return a ^ b;
  }

  constexpr friend std::string_view name(const Product&) {
    return "*";
  }

  friend std::ostream& operator<<(std::ostream& os, const Product& product) {
    return os << name(product);
  }
};

struct Inverse {
  constexpr Index outer(Index a, Index b) const {
    return a ^ b;
  }

  constexpr friend std::string_view name(const Inverse&) {
    return "/";
  }

  friend std::ostream& operator<<(std::ostream& os, const Inverse& inv) {
    return os << name(inv);
  }
};

struct Bind
{
  Index index;

  constexpr Bind(Index index) : index(index) {}

  /// The disjoint union representing the free indices.
  constexpr Index outer(Index a) const {
    return exclusive(a + index);
  }

  /// Search and replace in our stored index.
  constexpr void replace(Index is, Index with) {
    index.replace(is, with);
  }

  friend std::string name(const Bind& bind) {
    return "bind(" + std::string(name(bind.index)) + ")";
  }

  friend std::ostream& operator<<(std::ostream& os, const Bind& bind) {
    return os << name(bind);
  }
};

struct Partial
{
  Index index;

  constexpr Partial(Index index) : index(index) {}

  /// The disjoint union representing the free indices.
  constexpr Index outer(Index a) const {
    return exclusive(a + index);
  }

  /// Search and replace in our stored index.
  constexpr void replace(Index is, Index with) {
    index.replace(is, with);
  }

  constexpr void extend(Index i) {
    index += i;
  }

  friend std::string name(const Partial& dx) {
    return "dx(" + std::string(name(dx.index)) + ")";
  }

  friend std::ostream& operator<<(std::ostream& os, const Partial& dx) {
    return os << name(dx);
  }
};

struct Delta
{
  Index index;

  constexpr Delta(Index index) : index(index) {
    assert(size(index) == 2);
  }

  constexpr friend const Index& outer(const Delta& delta) {
    return delta.index;
  }

  constexpr std::tuple<Index> rebind(Index i) {
    assert(size(i) == 2);
    index = i;
    return std::tuple(index);
  }

  friend std::string name(const Delta& delta) {
    return "delta(" + std::string(name(delta.index)) + ")";
  }

  friend std::ostream& operator<<(std::ostream& os, const Delta& delta) {
    return os << name(delta);
  }
};

struct Zero {
  constexpr friend std::string_view name(const Zero&) {
    return "0";
  }

  friend std::ostream& operator<<(std::ostream& os, const Zero& zero) {
    return os << name(zero);
  }
};

struct One {
  constexpr friend std::string_view name(const One&) {
    return "1";
  }

  friend std::ostream& operator<<(std::ostream& os, const One& one) {
    return os << name(1);
  }
};

/// Node concepts.
template <typename T>
concept Binary =
 std::same_as<std::remove_cvref_t<T>, Sum> ||
 std::same_as<std::remove_cvref_t<T>, Difference> ||
 std::same_as<std::remove_cvref_t<T>, Product> ||
 std::same_as<std::remove_cvref_t<T>, Inverse>;

template <typename T>
concept Unary =
 std::same_as<std::remove_cvref_t<T>, Bind> ||
 std::same_as<std::remove_cvref_t<T>, Partial>;

template <typename T>
concept Leaf =
 std::same_as<std::remove_cvref_t<T>, Delta> ||
 std::same_as<std::remove_cvref_t<T>, Zero> ||
 std::same_as<std::remove_cvref_t<T>, One> ||
 std::same_as<std::remove_cvref_t<T>, TensorRef> ||
 std::same_as<std::remove_cvref_t<T>, Rational> ||
 std::same_as<std::remove_cvref_t<T>, double>;

/// Get the outer index for a binary node.
template <Binary Node>
constexpr Index outer(const Node& node, Index a, Index b) {
  return node.outer(a, b);
}

/// Rebind the indices in a subtree rooted in a binary node.
///
/// All of the binary nodes do the same thing to rebind their subtree. Using
/// the mapping between their own outer index and the rebind requested index,
/// `i`, they recursively rebind their children.
///
/// ```
/// auto tree0 = a(i,j) + b(j,i);
/// auto tree1 = tree0(u,v);
/// auto tree2 = a(u,v) + b(v,u);
/// assert(equivalent(tree1, tree2));
/// ```
///
/// In this case, the root of the tree is a sum with an outer index of `ij` and
/// it's being asked to rebind that to `uv`. It is told that its left child has
/// an outer index of `ij` and that its right child has an outer index of `ji`,
/// so it needs to forward the rebind request down the left subtree as `uv` and
/// down the right subtree as `vu`.
///
/// Multiplication and division are similar with the added complexity that there
/// will be a set of contracted indices *that are not rebound*, e.g., an MM
/// might have an outer index of `ij` and a contracted index of `k`, and so it
/// might forward `uk` to the left child and `kv` to the right child.
///
/// @tparam      Binary The binary node concept.
///
/// @param         node The binary node.
/// @param            i The index to which to rebind the tree.
/// @param            a The outer index of the left child.
/// @param            b The outer index of the right child.
///
/// @returns            The pair of outer indices that should be forwarded to
///                     rebind the children.
template <Binary Node>
constexpr std::tuple<Index, Index> rebind(Node& node, Index i, Index a, Index b) {
  Index o = node.outer(a, b);
  return std::tuple(a.replace(o, i), b.replace(o, i));
}

/// Get the outer index for a unary node.
template <Unary Node>
constexpr Index outer(const Node& n, Index a) {
  return n.outer(a);
}

/// Rebind the index in a unary node.
///
/// Both of our unary nodes have the same semantics with respect to their outer
/// inputs. They export the disjoint union of their stored index and their
/// child's outer index. This rebind request requires that we remap indices in
/// that disjoint union to the indices provided.
///
/// @tparam      Binary The unary node concept.
///
/// @param         node The unary node.
/// @param            i The index to which to rebind the tree.
/// @param            a The outer index of the node's child child.
///
/// @returns            The outer index that should be forwarded to rebind the
///                     child.
template <Unary Node>
constexpr std::tuple<Index> rebind(Node& n, Index i, Index a) {
  Index o = n.outer(a);
  n.replace(o, i);
  return std::tuple(a.replace(o, i));
}

constexpr std::tuple<Index> rebind(Delta& n, Index i) {
  return n.rebind(i);
}

/// These leaf nodes don't have outer indices, and thus have trivial rebind as
/// well.
constexpr Index outer(const Zero&)     { return {}; }
constexpr Index outer(const One&)      { return {}; }
constexpr Index outer(const Tensor&)   { return {}; }
constexpr Index outer(const Rational&) { return {}; }
constexpr Index outer(const double&)   { return {}; }

constexpr Index rebind(Zero&,         Index i) { assert(i.size() == 0); return {}; }
constexpr Index rebind(One&,          Index i) { assert(i.size() == 0); return {}; }
constexpr Index rebind(const Tensor&, Index i) { assert(i.size() == 0); return {}; }
constexpr Index rebind(Rational&,     Index i) { assert(i.size() == 0); return {}; }
constexpr Index rebind(double&,       Index i) { assert(i.size() == 0); return {}; }
}
