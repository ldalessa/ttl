#pragma once

#include "ParseTree.hpp"
#include "Tensor.hpp"
#include "concepts.hpp"

namespace ttl
{
  template <int M>
  struct Equation {
    using is_equation_tag = void;

    Tensor lhs;
    ParseTree<M> rhs;

    constexpr Equation(const Tensor& lhs, ParseTree<M> rhs)
        : lhs(lhs)
        , rhs(std::move(rhs))
    {
    }

    constexpr auto operator()(const auto& op) const
    {
      static_assert(requires { op(lhs, rhs); });
      return op(lhs, rhs);
    }
  };

  constexpr auto Tensor::operator<<=(is_parse_tree auto rhs) const {
    assert(order_ == rhs.outer().size());
    return Equation(*this, std::move(rhs));
  }
}
