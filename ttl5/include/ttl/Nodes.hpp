#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace ttl {
enum NodeType : char {
  SUM,
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

template <NodeType Type>
struct NodeTag {
  static constexpr NodeType value = Type;
};

template <NodeType Type>
constexpr inline NodeTag<Type> tag = {};

template <typename Tag>
concept LeafTag =
 std::same_as<std::remove_cvref_t<Tag>, NodeTag<DELTA>> ||
 std::same_as<std::remove_cvref_t<Tag>, NodeTag<TENSOR>> ||
 std::same_as<std::remove_cvref_t<Tag>, NodeTag<RATIONAL>> ||
 std::same_as<std::remove_cvref_t<Tag>, NodeTag<DOUBLE>>;

template <typename Tag>
concept UnaryTag =
 std::same_as<std::remove_cvref_t<Tag>, NodeTag<BIND>> ||
 std::same_as<std::remove_cvref_t<Tag>, NodeTag<PARTIAL>>;

template <typename Tag>
concept BinaryTag =
 std::same_as<std::remove_cvref_t<Tag>, NodeTag<SUM>> ||
 std::same_as<std::remove_cvref_t<Tag>, NodeTag<PRODUCT>> ||
 std::same_as<std::remove_cvref_t<Tag>, NodeTag<INVERSE>>;

template <typename Tag>
concept ValidTag =
 LeafTag<Tag> ||
 UnaryTag<Tag> ||
 BinaryTag<Tag>;

class Tensor;
using TensorRef = std::reference_wrapper<const Tensor>;

struct Sum {
  constexpr friend Index outer(const Sum&, Index a, Index b) {
    assert(permutation(a, b));
    return a;
  }

  friend std::ostream& operator<<(std::ostream& os, const Sum&) {
    return os << "+";
  }
};

struct Product {
  constexpr friend Index outer(const Product&, Index a, Index b) {
    return a ^ b;
  }

  friend std::ostream& operator<<(std::ostream& os, const Product&) {
    return os << "*";
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

struct Bind {
  Index index;

  constexpr Bind(Index index) : index(index) {}

  constexpr Index outer() const {
    return exclusive(index);
  }

  constexpr Index inner() const {
    return repeated(index);
  }

  constexpr Index all() const {
    return outer() + inner();
  }


  constexpr friend Index outer(const Bind& bind, Index) {
    return bind.outer();
  }

  friend std::ostream& operator<<(std::ostream& os, const Bind& bind) {
    return os << "bind(" << bind.all() << ")";
  }
};

struct Partial {
  Index index;

  constexpr Partial(Index index) : index(index) {}

  constexpr Index outer(Index a) const {
    return a ^ index;
  }

  constexpr friend Index outer(const Partial& dx, Index a) {
    return dx.outer(a);
  }

  friend std::ostream& operator<<(std::ostream& os, const Partial& dx) {
    return os << "dx(" << dx.index << ")";
  }
};

struct Delta {
  Index index;

  constexpr Delta(Index index) : index(index) {}

  constexpr friend Index outer(const Delta& delta) {
    return delta.index;
  }

  friend std::ostream& operator<<(std::ostream& os, const Delta& delta) {
    return os << "delta(" << delta.index << ")";
  }
};

constexpr Index outer(const Tensor&)   { return {}; }
constexpr Index outer(const Rational&) { return {}; }
constexpr Index outer(const double&)   { return {}; }

template <typename> constexpr inline NodeType type_of = INVALID;

template <> constexpr inline NodeType type_of<Sum>           = SUM;
template <> constexpr inline NodeType type_of<Product>       = PRODUCT;
template <> constexpr inline NodeType type_of<Inverse>       = INVERSE;
template <> constexpr inline NodeType type_of<Bind>          = BIND;
template <> constexpr inline NodeType type_of<Partial>       = PARTIAL;
template <> constexpr inline NodeType type_of<Delta>         = DELTA;
template <> constexpr inline NodeType type_of<const Tensor&> = TENSOR;
template <> constexpr inline NodeType type_of<TensorRef>     = TENSOR;
template <> constexpr inline NodeType type_of<Rational>      = RATIONAL;
template <> constexpr inline NodeType type_of<double>        = DOUBLE;

template <typename T>
concept Binary =
 std::same_as<std::remove_cvref_t<T>, Sum> ||
 std::same_as<std::remove_cvref_t<T>, Product> ||
 std::same_as<std::remove_cvref_t<T>, Inverse>;

template <typename T>
concept Unary =
 std::same_as<std::remove_cvref_t<T>, Bind> ||
 std::same_as<std::remove_cvref_t<T>, Partial>;

template <typename T>
concept Leaf =
 std::same_as<std::remove_cvref_t<T>, Delta> ||
 std::same_as<std::remove_cvref_t<T>, TensorRef> ||
 std::same_as<std::remove_cvref_t<T>, Rational> ||
 std::same_as<std::remove_cvref_t<T>, double>;
}
