#pragma once

#include <cstddef>
#include <format>
#include <ranges>
#include <string_view>

namespace ttl::parse
{
	struct index_view : std::string_view {
		using std::string_view::string_view;
	};

	static_assert(std::ranges::contiguous_range<index_view>);
}

template <>
struct std::formatter<ttl::parse::index_view> {
	static constexpr auto parse(auto& ctx)
	{
		return ctx.begin();
	}

	static constexpr auto format(ttl::parse::index_view index, auto& ctx)
	{
		auto out = ctx.out();
		if (index.size()) {
			out = std::format_to(out, "{}", index.front());
			index.remove_prefix(1);
			for (auto c : index) {
				out = std::format_to(out, ", {}", c);
			}
		}
		return out;
	}
};