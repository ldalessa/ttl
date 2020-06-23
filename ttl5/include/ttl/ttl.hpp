#pragma once

#include "Index.hpp"
#include "Rational.hpp"
#include "Tensor.hpp"
#include "Tree.hpp"

namespace ttl {
using index = Index;

template <char Id>
constexpr inline index idx = { Id };

using tensor = Tensor;

constexpr tensor scalar(std::string_view name) {
  return tensor(0, name);
}

constexpr tensor vector(std::string_view name) {
  return tensor(1, name);
}

using rational = Rational;

constexpr rational q(std::ptrdiff_t a) {
  return rational(a);
}

constexpr rational q(std::ptrdiff_t a, std::ptrdiff_t b) {
  return rational(a, b);
}
}
