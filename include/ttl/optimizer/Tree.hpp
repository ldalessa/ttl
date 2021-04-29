#pragma once

#include "ttl/Tensor.hpp"
#include "ttl/concepts.hpp"
#include "ttl/optimizer/ConstProp.hpp"
#include "ttl/optimizer/Dot.hpp"
#include "ttl/optimizer/LowerBinds.hpp"
#include "ttl/optimizer/Nodes.hpp"
#include "ttl/optimizer/Print.hpp"
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
          if (tag_is_binary(tag))
          {
            b = stack.pop_back();
            a = stack.pop_back();
          }
          else if (tag_is_unary(tag))
          {
            a = stack.pop_back();
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

      rhs_ = lower_binds(rhs_);
      rhs_ = const_prop(rhs_);
    }

    constexpr auto operator()(auto&& op) const
    {
      return op(lhs_, rhs_);
    }

    LowerBinds lower_binds = {};
    constexpr static ConstProp const_prop = {};

    auto print(FILE* file) const
    {
      Print print;
      print.format("{} = ", *lhs_);
      print(rhs_);
      print.format("\n");
      print.write(file);
    }

    auto dot(FILE* file) const
    {
      Dot dot;
      dot.format("graph {} {{\n", *lhs_);
      dot(rhs_);
      dot.format("}}\n");
      dot.write(file);
    }
  };
}
