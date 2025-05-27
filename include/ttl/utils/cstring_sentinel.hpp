/// Simple utility to allow cstrings to be used as std::ranges.
#pragma once

#include <cassert>
#include <ttl/concepts/character.hpp>

namespace ttl
{
	inline namespace utils
	{
		inline static struct CStringSentinal {
			template <concepts::character CharT>
			constexpr auto operator==(CharT const* str) const -> bool
			{
				assert(str);
				return *str == CharT();
			}
		} cstring_sentinel;
	}
}