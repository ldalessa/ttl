#pragma once

#include "Tensor.hpp"
#include "concepts.hpp"

namespace ttl
{
	template <is_tree Tree>
	struct Equation {
		using is_equation_tag = void;

		Tensor lhs;
		Tree rhs;

		constexpr Equation(const Tensor& lhs, Tree rhs)
			: lhs(lhs)
			, rhs(std::move(rhs))
		{
		}
	};

	template <is_tree Tree>
	constexpr auto Tensor::operator<<=(Tree&& rhs) const
	{
		assert(order_ == rhs.outer().size());
		return Equation(*this, std::forward<Tree>(rhs));
	}
}
