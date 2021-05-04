#pragma once

#include "ttl/concepts.hpp"
#include <bit>

namespace ttl
{
  using tag_id_t = unsigned;

  enum Tag : tag_id_t {
    SUM        = 1lu << 0,
    DIFFERENCE = 1lu << 1,
    PRODUCT    = 1lu << 2,
    RATIO      = 1lu << 3,
    POW        = 1lu << 4,
    BIND       = 1lu << 5,
    PARTIAL    = 1lu << 6,
    SQRT       = 1lu << 7,
    EXP        = 1lu << 8,
    NEGATE     = 1lu << 9,
    RATIONAL   = 1lu << 10,
    DOUBLE     = 1lu << 11,
    TENSOR     = 1lu << 12,
    SCALAR     = 1lu << 13,
    DELTA      = 1lu << 14,
    EPSILON    = 1lu << 15,
    MAX        = 1lu << 16
  };

  /// Tag operators (for building and testing tag sets).
  /// @{
  constexpr auto operator|(Tag a, Tag b) -> Tag
  {
    return Tag(tag_id_t(a) | tag_id_t(b));
  }

  constexpr auto operator&(Tag a, Tag b) -> Tag
  {
    return Tag(tag_id_t(a) & tag_id_t(b));
  }
  /// @}

  /// Tag sets... all of the sets contain MAX to disambiguate single-element
  /// sets from their single element.
  /// @{
  constexpr inline Tag NO_TAG         = Tag{0};
  constexpr inline Tag ALL            = Tag(MAX | MAX - 1);
  constexpr inline Tag ADDITION       = MAX | SUM | DIFFERENCE;
  constexpr inline Tag MULTIPLICATION = MAX | PRODUCT | RATIO;
  constexpr inline Tag BUILTIN2       = MAX | POW;
  constexpr inline Tag BINARY         = MAX | ADDITION | MULTIPLICATION | BUILTIN2;

  constexpr inline Tag BINDER         = MAX | BIND | PARTIAL;
  constexpr inline Tag BUILTIN1       = MAX | SQRT | EXP | NEGATE;
  constexpr inline Tag UNARY          = MAX | BINDER | BUILTIN1;

  constexpr inline Tag VARIABLE       = MAX | TENSOR | SCALAR;
  constexpr inline Tag IMMEDIATE      = MAX | RATIONAL | DOUBLE;
  constexpr inline Tag BUILTIN0       = MAX | DELTA | EPSILON;
  constexpr inline Tag CONSTANT       = MAX | IMMEDIATE | BUILTIN0;
  constexpr inline Tag LEAF           = MAX | CONSTANT | VARIABLE;
  /// @}

  template <Tag> struct tag_t;

  template <Tag tag>
  constexpr inline tag_t<tag> tag_v = {};

  template <> struct tag_t<ALL> {};

#define make_tag_t(child, parent)                   \
  template <> struct tag_t<child> : tag_t<parent> { \
    static constexpr Tag id = child;                \
  }

  /// Binary tag hierarchy
  /// @{
  make_tag_t(BINARY,         ALL);
  make_tag_t(ADDITION,       BINARY);
  make_tag_t(SUM,            ADDITION);
  make_tag_t(DIFFERENCE,     ADDITION);
  make_tag_t(MULTIPLICATION, BINARY);
  make_tag_t(PRODUCT,        MULTIPLICATION);
  make_tag_t(RATIO,          MULTIPLICATION);
  make_tag_t(BUILTIN2,       BINARY);
  make_tag_t(POW,            BUILTIN2);
  /// @}

  /// Unary tag hierarchy
  /// @{
  make_tag_t(UNARY,    ALL);
  make_tag_t(BINDER,   UNARY);
  make_tag_t(BIND,     BINDER);
  make_tag_t(PARTIAL,  BINDER);
  make_tag_t(BUILTIN1, UNARY);
  make_tag_t(SQRT,     BUILTIN1);
  make_tag_t(EXP,      BUILTIN1);
  make_tag_t(NEGATE,   BUILTIN1);
  /// @}

  /// Leaf tag hierarchy
  /// @{
  make_tag_t(LEAF,      ALL);
  make_tag_t(CONSTANT,  LEAF);
  make_tag_t(IMMEDIATE, CONSTANT);
  make_tag_t(RATIONAL,  IMMEDIATE);
  make_tag_t(DOUBLE,    IMMEDIATE);
  make_tag_t(BUILTIN0,  CONSTANT);
  make_tag_t(DELTA,     BUILTIN0);
  make_tag_t(EPSILON,   BUILTIN0);
  make_tag_t(VARIABLE,  LEAF);
  make_tag_t(TENSOR,    VARIABLE);
  make_tag_t(SCALAR,    VARIABLE);
  /// @}
#undef make_tag_t

  namespace tags
  {
    using all            = tag_t<ALL> const&;
    using binary         = tag_t<BINARY> const&;
    using addition       = tag_t<ADDITION> const&;
    using sum            = tag_t<SUM> const&;
    using difference     = tag_t<DIFFERENCE> const&;
    using multiplication = tag_t<MULTIPLICATION> const&;
    using product        = tag_t<PRODUCT> const&;
    using ratio          = tag_t<RATIO> const&;
    using builtin2       = tag_t<BUILTIN2> const&;
    using pow            = tag_t<POW> const&;
    using unary          = tag_t<UNARY> const&;
    using binder         = tag_t<BINDER> const&;
    using bind           = tag_t<BIND> const&;
    using partial        = tag_t<PARTIAL> const&;
    using builtin1       = tag_t<BUILTIN1> const&;
    using sqrt           = tag_t<SQRT> const&;
    using exp            = tag_t<EXP> const&;
    using negate         = tag_t<NEGATE> const&;
    using leaf           = tag_t<LEAF> const&;
    using constant       = tag_t<CONSTANT> const&;
    using immediate      = tag_t<IMMEDIATE> const&;
    using rational       = tag_t<RATIONAL> const&;
    using floating_point = tag_t<DOUBLE> const&;
    using builtin0       = tag_t<BUILTIN0> const&;
    using delta          = tag_t<DELTA> const&;
    using epsilon        = tag_t<EPSILON> const&;
    using variable       = tag_t<VARIABLE> const&;
    using tensor         = tag_t<TENSOR> const&;
    using scalar         = tag_t<SCALAR> const&;

    template <class T>
    concept is_binary = (T::id | BINARY) != NO_TAG;

    template <class T>
    concept is_unary = (T::id | UNARY) != NO_TAG;

    template <class T>
    concept is_leaf = (T::id | LEAF) != NO_TAG;
  }

  template <class Op, class... Args>
  constexpr auto visit(Tag tag, Op&& op, Args&&... args)
    -> decltype(auto)
  {
    switch (tag)
    {
     case SUM:        return std::forward<Op>(op)(tag_v<SUM>,        std::forward<Args>(args)...);
     case DIFFERENCE: return std::forward<Op>(op)(tag_v<DIFFERENCE>, std::forward<Args>(args)...);
     case PRODUCT:    return std::forward<Op>(op)(tag_v<PRODUCT>,    std::forward<Args>(args)...);
     case RATIO:      return std::forward<Op>(op)(tag_v<RATIO>,      std::forward<Args>(args)...);
     case BIND:       return std::forward<Op>(op)(tag_v<BIND>,       std::forward<Args>(args)...);
     case PARTIAL:    return std::forward<Op>(op)(tag_v<PARTIAL>,    std::forward<Args>(args)...);
     case POW:        return std::forward<Op>(op)(tag_v<POW>,        std::forward<Args>(args)...);
     case SQRT:       return std::forward<Op>(op)(tag_v<SQRT>,       std::forward<Args>(args)...);
     case EXP:        return std::forward<Op>(op)(tag_v<EXP>,        std::forward<Args>(args)...);
     case NEGATE:     return std::forward<Op>(op)(tag_v<NEGATE>,     std::forward<Args>(args)...);
     case RATIONAL:   return std::forward<Op>(op)(tag_v<RATIONAL>,   std::forward<Args>(args)...);
     case DOUBLE:     return std::forward<Op>(op)(tag_v<DOUBLE>,     std::forward<Args>(args)...);
     case TENSOR:     return std::forward<Op>(op)(tag_v<TENSOR>,     std::forward<Args>(args)...);
     case SCALAR:     return std::forward<Op>(op)(tag_v<SCALAR>,     std::forward<Args>(args)...);
     case DELTA:      return std::forward<Op>(op)(tag_v<DELTA>,      std::forward<Args>(args)...);
     case EPSILON:    return std::forward<Op>(op)(tag_v<EPSILON>,    std::forward<Args>(args)...);
     default:
      assert(false);
    };
    __builtin_unreachable();
  }

  // template <class Op, class... Args>
  // constexpr auto visit(Tag a, Tag b, auto const& op, Args&&... args)
  // {
  //   switch (b) {
  //    case SUM:        return visit(a, std::forward<Op>(op), tag_v<SUM>,        std::forward<Args>(args)...);
  //    case DIFFERENCE: return visit(a, std::forward<Op>(op), tag_v<Difference>, std::forward<Args>(args)...);
  //    case PRODUCT:    return visit(a, std::forward<Op>(op), tag_v<Product>,    std::forward<Args>(args)...);
  //    case RATIO:      return visit(a, std::forward<Op>(op), tag_v<Ratio>,      std::forward<Args>(args)...);
  //    case BIND:       return visit(a, std::forward<Op>(op), tag_v<Bind>,       std::forward<Args>(args)...);
  //    case PARTIAL:    return visit(a, std::forward<Op>(op), tag_v<Partial>,    std::forward<Args>(args)...);
  //    case POW:        return visit(a, std::forward<Op>(op), tag_v<Pow>,        std::forward<Args>(args)...);
  //    case SQRT:       return visit(a, std::forward<Op>(op), tag_v<Sqrt>,       std::forward<Args>(args)...);
  //    case EXP:        return visit(a, std::forward<Op>(op), tag_v<Exp>,        std::forward<Args>(args)...);
  //    case NEGATE:     return visit(a, std::forward<Op>(op), tag_v<Negate>,     std::forward<Args>(args)...);
  //    case RATIONAL:   return visit(a, std::forward<Op>(op), tag_v<RATIONAL>,   std::forward<Args>(args)...);
  //    case DOUBLE:     return visit(a, std::forward<Op>(op), tag_v<DOUBLE>,     std::forward<Args>(args)...);
  //    case TENSOR:     return visit(a, std::forward<Op>(op), tag_v<Tensor>,     std::forward<Args>(args)...);
  //    case SCALAR:     return visit(a, std::forward<Op>(op), tag_v<Scalar>,     std::forward<Args>(args)...);
  //    case DELTA:      return visit(a, std::forward<Op>(op), tag_v<Delta>,      std::forward<Args>(args)...);
  //    case EPSILON:    return visit(a, std::forward<Op>(op), tag_v<Epsilon>,    std::forward<Args>(args)...);
  //    default:
  //     assert(false);
  //   };
  //   __builtin_unreachable();
  // }

  constexpr auto ispow2(Tag tag)
  {
    return std::has_single_bit(tag_id_t(tag));
  }

  constexpr auto popcount(Tag tag)
  {
    return std::popcount(tag_id_t(tag));
  }

  constexpr auto ctz(Tag tag)
  {
    return std::countr_zero(tag_id_t(tag));
  }

  constexpr auto tag_is_simple(Tag tag)
  {
    return ispow2(tag);
  }

  constexpr auto tag_is_composite(Tag tag)
  {
    return popcount(tag) > 1;
  }

  constexpr auto tag_to_index(Tag tag) -> int
  {
    assert(tag_is_simple(tag));
    return ctz(tag);
  }

  constexpr bool tag_is(Tag tag, Tag set)
  {
    return tag_is_simple(tag & set);
  }

  constexpr bool tag_is_binary(Tag tag)
  {
    return tag_is(tag, BINARY);
  }

  constexpr bool tag_is_unary(Tag tag)
  {
    return tag_is(tag, UNARY);
  }

  constexpr bool tag_is_leaf(Tag tag)
  {
    return tag_is(tag, LEAF);
  }

  constexpr bool tag_is_immediate(Tag tag)
  {
    return tag_is(tag, IMMEDIATE);
  }

  constexpr bool tag_is_multiplication(Tag tag)
  {
    return tag_is(tag, MULTIPLICATION);
  }

  constexpr bool tag_is_addition(Tag tag)
  {
    return tag_is(tag, ADDITION);
  }

  constexpr bool tag_is_variable(Tag tag)
  {
    return tag_is(tag, VARIABLE);
  }

  template <class T>
  constexpr auto tag_outer(Tag tag, T const& a, T const& b = {}) -> T
  {
    if (tag_is_addition(tag)) {
      return a;
    }

    if (tag_is_multiplication(tag)) {
      return a ^ b;
    }

    if (tag == PARTIAL) {
      return exclusive(a + b);
    }

    if (tag == BIND) {
      return b;
    }

    return exclusive(a);
  }

  constexpr auto tag_eval(Tag tag, auto const& a, auto const& b)
  {
    assert(tag_is_binary(tag));

    using std::pow;
    switch (tag) {
     case SUM:        return a + b;
     case DIFFERENCE: return a - b;
     case PRODUCT:    return a * b;
     case RATIO:      return a / b;
     case POW:        return pow(a, b);
     default:
      assert(false);
    }
    __builtin_unreachable();
  }

  constexpr auto tag_to_string(Tag tag) -> const char*
  {
    constexpr const char* strings[] = {
      "+", // SUM
      "-", // DIFFERENCE
      "*", // PRODUCT
      "/", // RATIO
      "^", // POW
      "bind",  // BIND
      "∂", // PARTIAL
      "√", // SQRT
      "𝑒", // EXP
      "-", // NEGATE
      "",  // RATIONAL
      "",  // DOUBLE
      "",  // TENSOR
      "",  // SCALAR
      "δ", // DELTA
      "ε", // EPSILON
    };
    static_assert(std::size(strings) == tag_to_index(MAX));
    return strings[tag_to_index(tag)];
  }
}

#include <fmt/format.h>

template <>
struct fmt::formatter<ttl::Tag>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  constexpr auto format(ttl::Tag tag, auto& ctx)
  {
    return fmt::format_to(ctx.out(), "{}", tag_to_string(tag));
  }
};
