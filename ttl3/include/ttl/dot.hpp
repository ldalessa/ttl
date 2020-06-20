#pragma once

#include "Tree.hpp"
#include <ostream>
#include <string>
#include <string_view>

namespace ttl {
template <int M>
class TreeWriter {
  std::ostream& out_;
  const Tree<M>& tree_;

  template <typename T>
  std::string id(const T& node) const {
    return "node" + std::to_string(tree_.index_of(node));
  }

  template <mp::VariantType<Node> N>
  void edge(const N& from, const Node& to) {
    out_ << "\t" << id(from) << " -- " << id(to) << "[label=\"\"]\n";
  }

  void handle(const TensorRef& t) {
    out_ << "\t" << id(t) << "[label=\"" << name(t) << "[" << order(t) << "]" << "\"]\n";
  }

  void handle(const double& d) {
    out_ << "\t" << id(d) << "[label=\"" << d << "\"]\n";
  }

  void handle(const Rational& q) {
    out_ << "\t" << id(q) << "[label=\"" << q << "\"]\n";
  }

  void handle(const Delta& d) {
    out_ << "\t" << id(d) << "[label=\"delta(" << d.outer() << ")\"]\n";
  }

  void handle(const Bind& b) {
    out_ << "\t" << id(b) << "[label=\"bind(" << b.outer() << ")\"]\n";
    edge(b, tree_.right(b));
  }

  void handle(const Product& p) {
    out_ << "\t" << id(p) << "[label=\"*\"]\n";
    edge(p, tree_.left(p));
    edge(p, tree_.right(p));
  }

  void handle(const Sum& s) {
    out_ << "\t" << id(s) << "[label=\"+\"]\n";
    edge(s, tree_.left(s));
    edge(s, tree_.right(s));
  }

  void handle(const Partial& p) {
    out_ << "\t" << id(p) << "[label=\"dx(" << p.dx() << ")\"]\n";
    edge(p, tree_.right(p));
  }

  void handle(const Inverse& i) {
    out_ << "\t" << id(i) << "[label=\"/\"]\n";
    edge(i, tree_.left(i));
    edge(i, tree_.right(i));
  }

 public:
  TreeWriter(std::ostream& out, const Tree<M>& tree) : out_(out), tree_(tree) {
    for (const auto& node: tree) {
      std::visit([this](const auto& node) {
        handle(node);
      }, node);
    }
  }
};

class DotWriter {
  std::ostream& out_;
  std::string_view name_;

 public:
  explicit DotWriter(std::ostream& out, std::string_view name)
      : out_(out)
      , name_(name)
  {
  }

  template <int M>
  std::ostream& operator<<(const Tree<M>& tree) {
    out_ << "graph " << name_ << " {\n";
    TreeWriter _(out_, tree);
    out_ << "}\n";
    return out_;
  }
};

struct dot {
  std::string_view name;

  constexpr dot() = default;
  constexpr dot(std::string_view name) : name(name) {}
};

DotWriter operator<<(std::ostream& out, const dot& d) {
  return DotWriter(out, d.name);
}
}
