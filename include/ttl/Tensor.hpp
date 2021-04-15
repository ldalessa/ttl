#pragma once

#include "Index.hpp"
#include "concepts.hpp"
#include <string_view>
#include <fmt/core.h>

namespace ttl
{
  struct Tensor
  {
    std::string_view id_ = "";
    int order_ = -1;

    constexpr Tensor() = default;

    constexpr Tensor(std::string_view id, int order)
        : id_(id)
        , order_(order)
    {
    }

    constexpr friend bool operator==(Tensor const&, Tensor const&) = default;
    constexpr friend auto operator<=>(Tensor const&, Tensor const&) = default;

    constexpr auto order() const -> int
    {
      return order_;
    }

    constexpr auto id() const -> std::string_view
    {
      return id_;
    }

    // Implemented in Tree.hpp to avoid circular include.
    constexpr auto operator()(std::same_as<Index> auto... is) const;

    // Implemented in Equation to avoid circular include.
    constexpr auto operator=(is_tree auto&&) const;
  };

  constexpr auto to_string(const Tensor& t) -> std::string_view
  {
    return t.id();
  }

  constexpr auto scalar(std::string_view id) -> ttl::Tensor
  {
    return { id, 0 };
  }

  constexpr auto vector(std::string_view id) -> ttl::Tensor
  {
    return { id, 1 };
  }

  constexpr auto matrix(std::string_view id) -> ttl::Tensor
  {
    return { id, 2 };
  }
}

template <>
struct fmt::formatter<ttl::Tensor>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  constexpr auto format(const ttl::Tensor& tensor, auto& ctx)
  {
    return format_to(ctx.out(), "{}", to_string(tensor));
  }
};
