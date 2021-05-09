#pragma once

#include "ttl/Index.hpp"
#include "ttl/concepts.hpp"
#include <kumi.hpp>
#include <cassert>

namespace ttl
{
  template <kumi::product_type Equations>
  struct TensorExpression : TensorBase
  {
    using is_tensor_expression_tag = void;

    int rank_;
    Equations equations;

    constexpr TensorExpression(int rank, auto... eqns)
        : rank_     { rank }
        , equations { std::move(eqns)... }
    {
      assert(((ttl::rank(eqns) == 0) && ...));
    }

    constexpr auto rank() const -> int override
    {
      return rank_;
    }

    /// Bind a tensor with an index in an expression.
    ///
    /// The resulting tree can be captured as constexpr.
    ///
    /// Implemented in ParseTree.hpp to avoid circular include.
    constexpr auto bind_tensor(is_index auto... is) const;

    constexpr auto operator()(Index i, is_index auto... is) const
    {
      return bind_tensor(i, is...);
    }

    /// Bind a tensor with a scalar index.
    constexpr auto bind_scalar(std::signed_integral auto... is) const;

    constexpr auto operator()(std::signed_integral auto i, std::signed_integral auto... is) const
    {
      return bind_scalar(i, is...);
    }
  };

  TensorExpression(int, auto... eqns)
    -> TensorExpression<decltype(kumi::tuple{std::move(eqns)...})>;

  template <class... Args>
  requires((is_parse_tree<Args> || std::integral<Args>) && ...)
  constexpr auto scalar(Args... eqns)
  {
    return ttl::TensorExpression { 0, std::move(eqns)... };
  }

  template <class... Args>
  requires((is_parse_tree<Args> || std::integral<Args>) && ...)
  constexpr auto vector(Args... eqns)
  {
    return ttl::TensorExpression { 1, std::move(eqns)... };
  }

  template <class... Args>
  requires((is_parse_tree<Args> || std::integral<Args>) && ...)
  constexpr auto matrix(Args... eqns)
  {
    return ttl::TensorExpression { 2, std::move(eqns)... };
  }
}
