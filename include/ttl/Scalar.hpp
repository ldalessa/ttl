#pragma once

#include "ttl/Rational.hpp"
#include "ttl/ScalarIndex.hpp"
#include "ttl/Tag.hpp"
#include "ttl/Tensor.hpp"
#include "ttl/pow.hpp"
#include <kumi.hpp>
#include <cassert>
#include <bit>

namespace ttl
{
  struct Scalar
  {
    bool     constant;              //!< true if we think the scalar is constant
    int         order;              //!< the number of non-zero α
    int     direction;              //!< the non-zero α
    ScalarIndex     α;              //!< the ∂x_{i}^{α[i]} order values
    Tensor     tensor;              //!< the underlying tensor
    ScalarIndex index;              //!< the actual component

    constexpr Scalar() = default;

    constexpr Scalar(auto const* node)
        : Scalar(node->tensor, node->index, node->constant)
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
    /// @param t        The underlying tensor.
    /// @param incoming The specified index.
    /// @param constant True if the tensor is a constant.
    constexpr Scalar(Tensor const& t, ScalarIndex const& incoming, bool constant, int N = 0)
        : constant(constant)
        , order(0)
        , direction(0)
        , α()
        , tensor(t)
        , index(t.order())
    {
      assert(t.order() <= incoming.size());

      if (N) {
        α.resize(N);
      }

      int i = 0;
      for (; i < t.order(); ++i) {
        index[i] = incoming[i];
        assert(!N or incoming[i] < N);
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

    /// Check to see if the Scalar is valid in an N-dimensional expression.
    constexpr bool validate(int N)
    {
      if (α.size() > N) {
        return false;
      }
      α.ensure(N);
      for (auto i : index) {
        if (i >= N) {
          return false;
        }
      }
      return true;
    }

    constexpr auto operator=(std::floating_point auto d) const
    {
      return kumi::make_tuple(*this, double(d));
    }

    constexpr auto operator=(std::integral auto i) const
    {
      return kumi::make_tuple(*this, double(i));
    }

    constexpr auto operator=(Rational q) const
    {
      return kumi::make_tuple(*this, as<double>(q));
    }

    auto to_string() const -> std::string
    {
      constexpr static const char ids[] = { 'x', 'y', 'z', 'w' };

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

      assert(α.size());

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

  constexpr auto Tensor::bind_scalar(std::signed_integral auto... is) const
  {
    return Scalar(*this, { std::in_place, is... }, false);
  }

  constexpr auto Tensor::operator=(std::floating_point auto d) const
  {
    assert(order_ == 0);
    return bind_scalar() = d;
  }

  constexpr auto Tensor::operator=(std::integral auto i) const
  {
    assert(order_ == 0);
    return bind_scalar() = i;
  }

  constexpr auto Tensor::operator=(Rational q) const
  {
    assert(order_ == 0);
    return bind_scalar() = q;
  }
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
