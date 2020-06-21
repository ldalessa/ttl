#pragma once

#include "Tree.hpp"
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace ttl
{
class DotWriter
{
  std::ostream& out_;
  std::string_view name_;

  static std::string id(int i) {
    return "node" + std::to_string(i);
  }

  void edge(int parent, int child, Index a = {}) {
    out_ << "\t" << id(parent) << " -- " << id(child) << "[label=\"" << a << "\"]\n";
  }

 public:
  explicit DotWriter(std::ostream& out, std::string_view name)
      : out_(out)
      , name_(name)
  {
  }

  template <int M>
  std::ostream& operator<<(const Tree<M>& tree) && {
    out_ << "graph " << name_ << " {\n";
    visit(tree, [&](int i, const auto& node, auto&&... as) {
      out_ << "\t" << id(i) << "[label=\"" << node << "\"]\n";
      (edge(i, as.first), ...);
      return std::pair(i, outer(node, as.second...));
    });
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
