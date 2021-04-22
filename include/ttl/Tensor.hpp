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

    /// Bind a tensor with an index in an expression.
    ///
    /// The resulting tree can be captured as constexpr.
    ///
    /// Implemented in ParseTree.hpp to avoid circular include.
    constexpr auto bind_tensor(std::same_as<Index> auto... is) const;

    constexpr auto operator()(Index i, std::same_as<Index> auto... is) const
    {
      return bind_tensor(i, is...);
    }

    /// Bind a tensor with a scalar index.
    ///
    /// The resulting Scalar cannot be captured as a constexpr, but can be used
    /// inside of a constexpr expression context.
    ///
    /// This is implemented in Scalar.hpp to avoid circular include.
    constexpr auto bind_scalar(std::signed_integral auto... is) const;

    constexpr auto operator()(std::signed_integral auto i, std::signed_integral auto... is) const
    {
      return bind_scalar(i, is...);
    }

    constexpr auto operator=(std::floating_point auto d) const;
    constexpr auto operator=(std::integral auto i) const;
    constexpr auto operator=(Rational q) const;

    /// Create a differential equation of dt.
    ///
    /// This is implemented in Equation in order to avoid circular includes.
    constexpr auto operator<<=(is_tree auto&&) const;
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
