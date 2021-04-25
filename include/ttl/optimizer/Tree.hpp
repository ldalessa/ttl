#pragma once

#include "ttl/Tensor.hpp"
#include "ttl/concepts.hpp"
#include "ttl/optimizer/LowerBinds.hpp"
#include "ttl/optimizer/Nodes.hpp"
#include "ce/dvector.hpp"

namespace ttl::optimizer
{
  struct Tree
  {
    ttl::Tensor const* lhs_;
    node_ptr rhs_;

    constexpr Tree() = default;

    constexpr Tree(is_equation auto const& eqn)
    {
      eqn([&](auto const& lhs, is_parse_tree auto const& rhs)
      {
        lhs_ = &lhs;

        ce::dvector<node_ptr> stack; stack.reserve(rhs.depth());
        for (int i = 0, e = rhs.size(); i < e; ++i)
        {
          Tag tag = rhs.tag(i);
          node_ptr a;
          node_ptr b;
          if (tag.is_binary())
          {
            b = stack.pop_back();
            a = stack.pop_back();
          }
          else if (tag.is_unary())
          {
            b = stack.pop_back();
          }

          stack.emplace_back(
            tag,
            kw::a = std::move(a),
            kw::b = std::move(b),
            kw::d = rhs.ds[i],
            kw::q = rhs.qs[i],
            kw::tensor = rhs.tensors[i],
            kw::tensor_index = rhs.tensor_index[i],
            kw::scalar_index = rhs.scalar_index[i]);
        }
        assert(stack.size() == 1);
        rhs_ = stack.pop_back();
      });

      lower_binds();
    }

    constexpr auto operator()(auto&& op) const
    {
      return op(lhs_, rhs_);
    }

    constexpr void lower_binds()
    {
      LowerBinds lower;
      lower(rhs_);
    }
  };
}
