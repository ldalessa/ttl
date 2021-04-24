#pragma once

#include "ttl/concepts.hpp"

namespace ttl
{
  enum TagIDs : int {
    SUM = 0,
    DIFFERENCE,
    PRODUCT,
    RATIO,
    POW,
    BIND,
    PARTIAL,
    SQRT,
    EXP,
    NEGATE,
    RATIONAL,
    DOUBLE,
    TENSOR,
    SCALAR,
    DELTA,
    EPSILON,
    MAX
  };

  struct Tag
  {
    TagIDs id_ = MAX;

    constexpr Tag() = default;

    constexpr Tag(TagIDs id)
        : id_ { id }
    {
    }

    constexpr Tag& operator=(TagIDs id)
    {
      id_ = id;
      return *this;
    }

    constexpr friend bool operator==(Tag, Tag) = default;
    constexpr friend auto operator<=>(Tag, Tag) = default;

    constexpr friend bool operator==(Tag tag, TagIDs id)
    {
      return tag.id_ == id;
    }

    constexpr friend bool operator==(TagIDs id, Tag tag)
    {
      return id == tag.id_;
    }

    constexpr operator TagIDs() const
    {
      return id_;
    }

    constexpr bool is_binary() const
    {
      return id_ <= POW;
    }

    constexpr bool is_unary() const
    {
      return not is_binary() && id_ <= NEGATE;
    }

    template <is_index T>
    constexpr auto outer(T const& a, T const& b = {}) const -> T
    {
      if (id_ == SUM or id_ == DIFFERENCE) {
        return a;
      }

      if (id_ == PRODUCT or id_ == RATIO) {
        return a ^ b;
      }

      if (id_ == PARTIAL) {
        return exclusive(a + b);
      }

      if (id_ == BIND) {
        return b;
      }

      return exclusive(a);
    }

    constexpr auto to_string() const -> const char*
    {
      constexpr const char* strings[] = {
        "+", // SUM
        "-", // DIFFERENCE
        "*", // PRODUCT
        "/", // RATIO
        "^", // POW
        "bind",  // BIND
        "∂", // PARTIAL
        "√", // SQRT
        "ℇ", // EXP
        "-", // NEGATE
        "",  // RATIONAL
        "",  // DOUBLE
        "",  // TENSOR
        "",  // SCALAR
        "δ", // DELTA
        "ε", // EPSILON
      };
      static_assert(std::size(strings) == MAX);
      return strings[id_];
    }

    constexpr friend auto to_string(Tag tag) -> const char*
    {
      return tag.to_string();
    }
  };
}

#include <fmt/format.h>

template <>
struct fmt::formatter<ttl::Tag>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  constexpr auto format(ttl::Tag tag, auto& ctx)
  {
    return fmt::format_to(ctx.out(), "{}", tag.to_string());
  }
};
