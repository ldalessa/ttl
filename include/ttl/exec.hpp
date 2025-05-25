#pragma once

#include "ttl/ScalarIndex.hpp"
#include "ttl/pow.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <utility>

namespace ttl::exec
{
	enum Tag : int {
		SUM,
		DIFFERENCE,
		PRODUCT,
		RATIO,
		IMMEDIATE,
		SCALAR,
		CONSTANT,
		DELTA
	};

	constexpr bool is_binary(Tag tag)
	{
		return tag < IMMEDIATE;
	}

	struct Index {
		const char* i;
		const char* e;

		constexpr friend bool operator==(Index const& a, Index const& b)
		{
			return (a.size() == b.size()) && std::equal(a.i, a.e, b.i, b.e);
		}

		constexpr friend auto operator<=>(Index const& a, Index const& b)
		{
			return std::lexicographical_compare_three_way(a.i, a.e, b.i, b.e);
		}

		constexpr auto operator[](int index) const -> char
		{
			return i[index];
		}

		constexpr auto size() const -> int
		{
			return e - i;
		}

		constexpr auto index_of(char c) const -> int
		{
			for (auto ii = i; ii < e; ++ii) {
				if (*ii == c) {
					return ii - i;
				}
			}
			std::unreachable();
		}
	};

	template <int N, int M>
	constexpr auto make_map(Index const& from, Index const& to)
		-> std::array<int, ttl::pow(N, M)>
	{
		assert(from.size() == M);
		std::array<int, ttl::pow(N, M)> out;
		ScalarIndex index(from.size());
		int i = 0;
		do {
			out[i++] = index.select(from, to).row_major(N);
		} while (index.carry_sum_inc(N));
		return out;
	}

} // namespace exec
