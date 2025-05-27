#pragma once

#include <cassert>
#include <format>

namespace ttl
{
	struct index {
		char c;

		consteval index(char c)
			: c(c) { }

		consteval index(char const (&str)[2])
			: c(str[0])
		{
			assert(str[1] == '\0');
		}
	};
}

template <>
struct std::formatter<ttl::index> {
	static constexpr auto parse(auto& ctx)
	{
		return ctx.begin();
	}

	static constexpr auto format(ttl::index index, auto& ctx)
	{
		return std::format(index.c, ctx.out());
	}
};