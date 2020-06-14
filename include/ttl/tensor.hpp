#pragma once

#include "index.hpp"
#include <string_view>

namespace ttl
{
template <int N> class bind;

class tensor {
  int order_;
  std::string_view name_;
 public:
  constexpr tensor(const tensor&) = delete;
  constexpr tensor(tensor&&) = delete;

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

  template <Index... Is>
  constexpr decltype(auto) operator()(Is... is) const;
};
}
