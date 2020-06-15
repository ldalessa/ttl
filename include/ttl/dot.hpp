#pragma once

#include "index.hpp"
#include "nodes.hpp"
#include <ostream>
#include <tuple>

namespace ttl {
class dot_writer {
  std::ostream& out_;
  std::string_view name_;
  int i_ = 0;

  /// Print a leaf tensor node (we'd like to know its rank).
  auto node(const tensor& t) {
    out_ << "\tnode" << i_ << "[label=\"" << name(t) << "[" << order(t) << "]" << "\"]\n";
    return std::tuple(i_++, index());
  }

  template <Expression Expr>
  auto node(Expr&& expr) {
    out_ << "\tnode" << i_ << "[label=\"" << name(expr) << "\"]\n";
    return std::tuple(i_++, index());
  }

/// Print the children expressions, each of which will return its node id and
  /// outer index, and then print this node and link to the children.
  template <Internal Expr>
  auto node(Expr&& expr) {
    std::tuple cs = std::apply([&](auto&&... cs) {
      return std::make_tuple(node(std::forward<decltype(cs)>(cs))...);
    }, children(expr));

    out_ << "\tnode" << i_ << "[label=\"" << name(expr) << "\"]\n";

    auto edge = [this](int j, auto is) {
      out_ << "\tnode" << i_ << " -- node" << j << "[label=\"";
      for (auto&& c : is) {
        out_ << c;
      }
      out_ << "\"]\n";
    };

    std::apply([&](auto... c) {
      (std::apply(edge, c), ...);
    }, cs);

    return std::tuple(i_++, outer(expr));
  }

 public:
  explicit dot_writer(std::ostream& out, std::string_view name)
      : out_(out)
      , name_(name)
  {
  }

  template <Expression Expr>
  friend std::ostream& operator<<(dot_writer&& dot, Expr&& expr) {
    dot.out_ << "graph " << dot.name_ << " {\n";
    dot.node(std::forward<Expr>(expr));
    dot.out_ << "}\n";
    return dot.out_;
  }
};

class dot {
  std::string_view name_;

 public:
  explicit dot(std::string_view name) : name_(name) {}

  friend dot_writer operator<<(std::ostream& out, const dot& d) {
    return dot_writer{ out, d.name_ };
  }
};
}
