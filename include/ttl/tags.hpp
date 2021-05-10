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
