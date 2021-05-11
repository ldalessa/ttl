#pragma once

namespace ttl
{
  enum TreeTag : int {
    SUM,
    DIFFERENCE,
    PRODUCT,
    RATIO,
    BIND,
    NEGATE,
    EXPONENT,
    PARTIAL,
    CMATH,
    LITERAL,
    TENSOR,
    SCALAR,
    DELTA,
    EPSILON
  };

  constexpr inline bool is_binary(TreeTag tag)
  {
    return tag < NEGATE;
  }

  constexpr inline bool is_unary(TreeTag tag)
  {
    return NEGATE <= tag and tag < LITERAL;
  }

  constexpr inline bool is_leaf(TreeTag tag)
  {
    return LITERAL <= tag;
  }

  enum CMathTag : int {
    ABS,
    FMIN,
    FMAX,
    EXP,
    LOG,
    POW,
    SQRT,
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    ATAN2,
    SINH,
    COSH,
    TANH,
    ASINH,
    ACOSH,
    ATANH,
    CEIL,
    FLOOR
  };

}
