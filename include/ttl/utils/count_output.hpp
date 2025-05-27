#pragma once

#include <cstddef>
#include <iterator>
#include <utility>

namespace ttl
{
	inline namespace utils
	{
		struct counting_output_iterator {
			using difference_type = std::ptrdiff_t;

			std::size_t _count {};

			constexpr auto operator*() const
			{
				return std::ignore;
			}

			constexpr auto operator++(int) -> counting_output_iterator
			{
				return counting_output_iterator {
					._count = _count++
				};
			}

			constexpr auto operator++() -> counting_output_iterator&
			{
				_count++;
				return *this;
			}

			constexpr operator std::size_t() const
			{
				return _count;
			}
		};

		static_assert(std::output_iterator<counting_output_iterator, int>);

		static constexpr counting_output_iterator count_output {};
	}
}