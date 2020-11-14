#pragma once

namespace ttl {
struct ScalarRPNTreeNode {
  enum Tag {
    SUM,
    DIFFERENCE,
    PRODUCT,
    RATIO,
    DOUBLE,
    RATIONAL,
    CONSTANT,
    SCALAR
  };

  Tag  tag;
  int left;
  union {
    struct {} _monostate = {};
    int offset;
    Rational q;
    double   d;
  }
};

template <int M>
struct ScalarRPNTree {
  ScalarRPNTreeNode nodes[M];
}
}
