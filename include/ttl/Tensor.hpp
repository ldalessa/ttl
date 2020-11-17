#pragma once

#include "Index.hpp"
#include "concepts.hpp"
#include <string_view>
#include <fmt/core.h>

namespace ttl {
struct Tensor
{
 private:
  std::string_view id_;
  int order_;

 public:
  constexpr Tensor() = default;

  constexpr Tensor(std::string_view id, int order) : id_(id), order_(order) {
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

constexpr std::string_view to_string(const Tensor& t) {
  return t.id();
}

constexpr ttl::Tensor scalar(std::string_view id) {
  return { id, 0 };
}

constexpr ttl::Tensor vector(std::string_view id) {
  return { id, 1 };
}
}

template <>
struct fmt::formatter<ttl::Tensor> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::Tensor& tensor, FormatContext& ctx) {
    return format_to(ctx.out(), "{}", to_string(tensor));
  }
};
