#pragma once

#include "Index.hpp"
#include "Nodes.hpp"
#include "Tree.hpp"
#include <ostream>
#include <string_view>

namespace ttl {
class Tensor {
  int order_;
  std::string_view name_;

 public:
  constexpr Tensor() = delete;
  constexpr Tensor(const Tensor&) = delete;
  constexpr Tensor(Tensor&&) = delete;

  constexpr Tensor(int order, std::string_view name) : order_(order), name_(name) {
  }

  constexpr int order() const {
    return order_;
  }

  constexpr std::string_view name() const {
    return name_;
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& a) {
    return os << a.name_ << "(" << a.order_ << ")";
  }

  constexpr auto operator()(IsIndex auto... is) const {
    return make_tree(Bind((is + ... + Index())), make_tree(std::cref(*this)));
  }
};

constexpr int order(const Tensor& t) {
  return t.order();
}

constexpr std::string_view name(const Tensor& t) {
  return t.name();
}
}
