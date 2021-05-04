#pragma once

#include "ttl/Tensor.hpp"
#include "ttl/concepts.hpp"
#include "ttl/optimizer/ConstProp.hpp"
#include "ttl/optimizer/Dot.hpp"
#include "ttl/optimizer/LowerBinds.hpp"
#include "ttl/optimizer/LowerPartials.hpp"
#include "ttl/optimizer/Nodes.hpp"
#include "ttl/optimizer/Print.hpp"
#include "ttl/optimizer/Simplify.hpp"
#include "ce/dvector.hpp"

namespace ttl::optimizer
{
  struct Tree
  {
    ttl::Tensor const* lhs_;
    node_ptr rhs_;

    constexpr static ConstProp         const_prop = {};
    constexpr static LowerBinds       lower_binds = {};
    constexpr static LowerPartials lower_partials = {};
    constexpr static Simplify            simplify = {};

    constexpr Tree() = default;

    constexpr Tree(is_equation auto const& eqn, auto const& constants)
    {
      eqn([&](Tensor const* lhs, is_parse_tree auto const& rhs)
      {
        lhs_ = lhs;

        ce::dvector<bool> constant; constant.reserve(rhs.depth());
        ce::dvector<node_ptr> nodes; nodes.reserve(rhs.depth());
        for (int i = 0, e = rhs.size(); i < e; ++i)
        {
          Tag tag = rhs.tag(i);
          node_ptr a;
          node_ptr b;
          bool c;

          if (tag_is_binary(tag))
          {
            b = nodes.pop_back();
            a = nodes.pop_back();
            constant.push_back(constant.pop_back() && constant.pop_back());
          }
          else if (tag_is_unary(tag))
          {
            a = nodes.pop_back();
          }
          else if (tag_is_variable(tag))
          {
            constant.push_back(constants(rhs.tensors[i]));
          }
          else {
            constant.push_back(true);
          }

          nodes.emplace_back(
              new Node(
                  tag,
                  std::move(a),
                  std::move(b),
                  rhs.qs[i],
                  rhs.ds[i],
                  rhs.tensors[i],
                  rhs.tensor_index[i],
                  rhs.scalar_index[i],
                  constant.back()));
        }
        assert(nodes.size() == 1);
        rhs_ = nodes.pop_back();
      });

      rhs_ = lower_binds(rhs_);
      rhs_ = const_prop(rhs_);
      rhs_ = simplify(rhs_);
      rhs_ = lower_partials(rhs_);
      rhs_ = const_prop(rhs_);
      rhs_ = simplify(rhs_);
    }

    constexpr auto operator()(auto&& op) const
    {
      return op(lhs_, rhs_);
    }

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
