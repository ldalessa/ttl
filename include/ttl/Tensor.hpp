#pragma once

#include <format>
#include <ttl/expect.hpp>
#include <ttl/index.hpp>

namespace ttl
{
	namespace parse
	{
		template <std::size_t, std::size_t>
		struct Tree;
	}

	struct tensor {
		std::string_view _id;
		std::size_t _rank;

		constexpr auto rank() const -> std::size_t
		{
			return _rank;
		}

		consteval auto operator()(auto... is) const -> parse::Tree<1, sizeof...(is)>
		{
			expect(sizeof...(is) == _rank);
			return _bind(index(is)...);
		}

	private:
		// implemented in parse/tree.hpp
		consteval auto _bind(auto... is) const -> parse::Tree<1, sizeof...(is)>;
	};

	consteval auto scalar(char const* id) -> ttl::tensor
	{
		return { id, 0 };
	}

	consteval auto vector(char const* id) -> ttl::tensor
	{
		return { id, 1 };
	}

	consteval auto matrix(char const* id) -> ttl::tensor
	{
		return { id, 2 };
	}
}

template <>
struct std::formatter<ttl::tensor> {
	static constexpr auto parse(auto& ctx)
	{
		return ctx.begin();
	}

	static constexpr auto format(ttl::tensor const& a, auto& ctx)
	{
		return std::format_to(ctx.out(), "{}", a._id);
	}
};