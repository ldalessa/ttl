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

    auto print(FILE* out) const
    {
      fmt::print(out, "{} = {}\n", lhs, to_string(*rhs));
    }

    auto dot(FILE* file) const
    {
      fmt::memory_buffer out;
      fmt::format_to(out, "graph {} {{\n", lhs);
      rhs.to_dot(out);
      fmt::format_to(out, "{}", "}}\n");
      std::fwrite(out.data(), out.size(), 1, file);
    }
  };

  constexpr auto Tensor::operator<<=(is_parse_tree auto rhs) const {
    assert(order_ == rhs.outer().size());
    return Equation(*this, std::move(rhs));
  }
}
