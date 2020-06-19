#pragma once

#include "index.hpp"
#include <string_view>
#include <tuple>

namespace ttl
{
template <Index> class bind;

class tensor {
  int order_;
  std::string_view name_;
 public:
  constexpr tensor(const tensor&) = delete;
  constexpr tensor(tensor&&) = delete;
  constexpr tensor& operator=(const tensor&) = delete;
  constexpr tensor& operator=(tensor&&) = delete;

  constexpr tensor(int order, std::string_view name)
      : order_(order),
        name_(name)
  {
  }

  friend constexpr int order(const tensor& t) {
    return t.order_;
  }

  friend constexpr std::string_view name(const tensor& t) {
    return t.name_;
  }

  friend constexpr std::tuple<> children(const tensor&) {
    return {};
  }

  template <Index... Is>
  constexpr decltype(auto) operator()(Is... is) const;
};

constexpr tensor scalar(std::string_view name) {
  return { 0, name };
}

constexpr tensor vector(std::string_view name) {
  return { 1, name };
}
}
