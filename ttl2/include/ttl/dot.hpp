#pragma once

#include "tree.hpp"
#include <ostream>
#include <vector>

namespace ttl {
class dot_writer {
  std::ostream& out_;
  std::string_view name_;
  int i_ = 0;
  std::vector<std::pair<int, index>> data_;

  void edge(int to, index i) {
    out_ << "\tnode" << i_ << " -- node" << to << "[label=\"" << name(i) << "\"]\n";
  }

  void handle(const tensor& t) {
    out_ << "\tnode" << i_ << "[label=\"" << name(t) << "[" << order(t) << "]" << "\"]\n";
    data_.emplace_back(i_++, index());
  }

  void handle(index i) {
    data_.emplace_back(i_, i);
  }

  void handle(double d) {
    out_ << "\tnode" << i_ << "[label=\"" << d << "\"]\n";
    data_.emplace_back(i_++, index());
  }

  void handle(rational q) {
    out_ << "\tnode" << i_ << "[label=\"" << q.p << "/" << q.q << "\"]\n";
    data_.emplace_back(i_++, outer(q));
  }

  void handle(Delta d) {
    out_ << "\tnode" << i_ << "[label=\"delta(" << name(outer(d)) << ")\"]\n";
    data_.emplace_back(i_++, outer(d));
  }

  void handle(Bind b) {
    auto [ri, r] = data_.back(); data_.pop_back();
    auto [li, l] = data_.back(); data_.pop_back();
    edge(li, l);
    out_ << "\tnode" << i_ << "[label=\"bind(" << name(l) << "=" << name(r) << ")\"]\n";
    data_.emplace_back(i_++, outer(b, l, r));
  }

  void handle(Product p) {
    out_ << "\tnode" << i_ << "[label=\"*\"]\n";
    auto [ri, r] = data_.back(); data_.pop_back();
    auto [li, l] = data_.back(); data_.pop_back();
    edge(ri, r);
    edge(li, l);
    data_.emplace_back(i_++, outer(p, l, r));
  }

  void handle(Sum s) {
    out_ << "\tnode" << i_ << "[label=\"+\"]\n";
    auto [ri, r] = data_.back(); data_.pop_back();
    auto [li, l] = data_.back(); data_.pop_back();
    edge(ri, r);
    edge(li, l);
    data_.emplace_back(i_++, outer(s, l, r));
  }

  void handle(Partial p) {
    auto [ri, r] = data_.back(); data_.pop_back();
    auto [li, l] = data_.back(); data_.pop_back();
    out_ << "\tnode" << i_ << "[label=\"dx(" << name(r) << ")\"]\n";
    edge(li, l);
    data_.emplace_back(i_++, outer(p, l, r));
  }

  void handle(Inverse i) {
    out_ << "\tnode" << i_ << "[label=\"/\"]\n";
    auto [ri, r] = data_.back(); data_.pop_back();
    auto [li, l] = data_.back(); data_.pop_back();
    edge(ri, r);
    edge(li, l);
    data_.emplace_back(i_++, outer(i, l, r));
  }

 public:
  explicit dot_writer(std::ostream& out, std::string_view name)
      : out_(out)
      , name_(name)
  {
  }

  template <int M>
  std::ostream& operator<<(const Tree<M>& tree) {
    assert(tree.size() > 2);

    out_ << "graph " << name_ << " {\n";

    for (auto&& n : tree) {
      std::visit([&](auto&& node) {
        handle(std::forward<decltype(node)>(node));
      }, n);
    }

    out_ << "}\n";
    return out_;
  }
};

struct dot {
  std::string_view name;

  constexpr dot() = default;
  constexpr dot(std::string_view name) : name(name) {}
};

dot_writer operator<<(std::ostream& out, const dot& d) {
  return dot_writer{ out, d.name };
}
}
