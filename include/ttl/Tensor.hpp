#pragma once

#include "Index.hpp"
#include "concepts.hpp"

#include <fmt/format.h>
#include <string_view>

namespace ttl {
struct Tensor
{
 private:
  int order_;
  std::string_view id_;

 public:
  constexpr Tensor(int order, std::string_view id) : order_(order), id_(id) {
  }

  constexpr int order() const {
    return order_;
  }

  constexpr std::string_view id() const {
    return id_;
  }

  // Implemented in Tree.hpp to avoid circular include.
  constexpr auto operator()(std::same_as<Index> auto... is) const;

  constexpr bool operator==(const Tensor& b) const {
    return id_ == b.id_;
  }

  constexpr auto operator<=>(const Tensor& b) const {
    return id_ <=> b.id_;
  }

  // Implemented in Equation to avoid circular include.
  constexpr auto operator=(is_tree auto&&) const;
};

constexpr ttl::Tensor scalar(std::string_view id) {
  return { 0, id };
}

constexpr ttl::Tensor vector(std::string_view id) {
  return { 1, id };
}
}

template <>
struct fmt::formatter<ttl::Tensor> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const ttl::Tensor& tensor, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", tensor.id());
  }
};
