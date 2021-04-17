#pragma once

#include "ScalarIndex.hpp"
#include "Tag.hpp"
#include "Tensor.hpp"
#include "pow.hpp"
#include <cassert>
#include <bit>

namespace ttl
{
  struct Scalar
  {
    bool     constant;                      //
    int         order;                      //!< the number of non-zero α
    int     direction;                      //!< the non-zero α
    ScalarIndex     α;                      //!< the ∂x_{i}^{α[i]} order values
    Tensor     tensor;                      //!< the underlying tensor
    int     component;                      //!< the linearized scalar component
    ScalarIndex index;                      //!< the actual component

    constexpr Scalar() = default;

    constexpr Scalar(int N, auto const* node)
        : Scalar(N, node->tensor, node->index, node->constant)
    {
      assert(node->tag == TENSOR);
    }

    /// Manifest a scalar for a particular tensor index.
    ///
    /// Tensors can be indexed by arrays of integers. When this happens, the
    /// user (or tree) is trying to refer to a specific scalar in a tensor
    /// derived from the underlying tensor. The first `t.order()` indices refer
    /// to the actual scalar in the tensor, and the rest produce a specific
    /// partial derivative.
    ///
    ///    Tensor t = vector(v);
    ///    Scalar s = t(2, 0, 0, 1, 1) // ∂v_{z}/∂x^2y^2
    ///
    /// @param N        The dimensionality (we need to know this independently).
    /// @param t        The underlying tensor.
    /// @param incoming The specified index.
    /// @param constant True if the tensor is a constant.
    constexpr Scalar(int N, Tensor const& t, ScalarIndex const& incoming, bool constant)
        : constant(constant)
        , order(0)
        , direction(0)
        , α(N)
        , tensor(t)
        , component(0)
        , index(t.order())
    {
      assert(t.order() <= incoming.size());

      // Runtime constant coefficients may be provided for scalars that are
      // impossible for the problem dimensionality. Just set some "impossible"
      // value for the component if we see one of those.
      for (int i : incoming) {
        if (N <= i) {
          component = -1;
          return;
        }
      }

      int i = 0;
      for (; i < t.order(); ++i) {
        index[i] = incoming[i];
        component += ttl::pow(N, i) * incoming[i];
      }

      for (; i < incoming.size(); ++i) {
        α[incoming[i]] += 1;
        direction |= ttl::pow(2, incoming[i]);
      }

      order = std::popcount(unsigned(direction));

      assert(!constant || direction == 0);
    }

    constexpr friend bool operator==(Scalar const&, Scalar const&) = default;
    constexpr friend auto operator<=>(Scalar const&, Scalar const&) = default;

    auto to_string() const -> std::string
    {
      constexpr static const char ids[] = { 'x', 'y', 'z', 'w' };

      int N = α.size();

      std::string str;

      if (order != 0) {
        str.append("∂");
      }

      str.append(tensor.id());

      for (int i : index) {
        assert(i < std::size(ids));
        str.append(1, ids[i]);
      }

      if (direction == 0) {
        return str;
      }

      str.append("_∂");

      for (int n = 0; n < α.size(); ++n) {
        for (int i = 0; i < α[n]; ++i) {
          assert(i < std::size(ids));
          str.append(1, ids[n]);
        }
      }
      return str;
    }
  };
}

#include <fmt/format.h>

template <>
struct fmt::formatter<ttl::Scalar>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  auto format(ttl::Scalar const& p, auto& ctx)
  {
    return format_to(ctx.out(), "{}", p.to_string());
  }
};
