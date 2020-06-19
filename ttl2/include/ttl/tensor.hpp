#pragma once

#include <concepts>
#include <string_view>

namespace ttl {
template <int M> class Tree;

class tensor {
  int order_;
  std::string_view name_;

 public:
  constexpr tensor() = delete;
  constexpr tensor(const tensor&) = delete;
  constexpr tensor(tensor&&) = delete;

  constexpr tensor(int order, std::string_view name) : order_(order), name_(name) {
  }

  constexpr int order() const {
    return order_;
  }

  constexpr std::string_view name() const {
    return name_;
  }

  template <Index... Is>
  constexpr Tree<3> operator()(Is...) const;
};

constexpr std::string_view name(const tensor& t) {
  return t.name();
}

constexpr int order(const tensor& t) {
  return t.order();
}

template <typename T>
concept Tensor = std::same_as<tensor, std::remove_cvref_t<T>>;

constexpr ttl::tensor scalar(std::string_view name) {
  return { 0, name };
}

constexpr ttl::tensor vector(std::string_view name) {
  return { 1, name };
}
}
