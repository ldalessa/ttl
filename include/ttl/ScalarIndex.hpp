#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <format>

namespace ttl
{
	struct ScalarIndex {
		int size_ = 0;
		int data_[TTL_MAX_PARSE_INDEX] = {};

		constexpr ScalarIndex() = default;

		constexpr ScalarIndex(int n)
			: size_(n)
		{
			assert(0 <= n and (unsigned) n < std::size(data_));
		}

		constexpr ScalarIndex(std::in_place_t, std::signed_integral auto... is)
			: size_ { sizeof...(is) }
			, data_ { int(is)... }
		{
			static_assert(sizeof...(is) < TTL_MAX_PARSE_INDEX);
			assert(((0 <= is) && ...));
		}

		constexpr friend bool operator==(ScalarIndex const& a, ScalarIndex const& b)
		{
			return (a.size_ == b.size_) && std::equal(std::begin(a.data_), std::begin(a.data_) + a.size_, std::begin(b.data_), std::begin(b.data_) + b.size_);
		}

		constexpr friend auto operator<=>(ScalarIndex const& a, ScalarIndex const& b)
		{
			return std::lexicographical_compare_three_way(
				std::begin(a.data_), std::begin(a.data_) + a.size_,
				std::begin(b.data_), std::begin(b.data_) + b.size_);
		}

		constexpr auto size() const -> decltype(auto)
		{
			return size_;
		}

		constexpr auto begin() const -> decltype(auto)
		{
			return std::begin(data_);
		}

		constexpr auto end() const -> decltype(auto)
		{
			return begin() + size();
		}

		constexpr auto operator[](int i) const -> decltype(auto)
		{
			return data_[i];
		}

		constexpr auto operator[](int i) -> decltype(auto)
		{
			return data_[i];
		}

		constexpr void resize(int n)
		{
			size_ = n;
		}

		constexpr void ensure(int n)
		{
			size_ = std::max(size_, n);
		}

		constexpr auto row_major(int N) const
		{
			int sum = 0;
			int n = 1;
			for (int i = 0; i < size(); ++i) {
				sum += n * data_[i];
				n *= N;
			}
			return sum;
		}

		constexpr auto select(auto&& from, auto&& to) const
			-> ScalarIndex
		{
			assert(size_ == from.size());
			ScalarIndex out(to.size());
			for (int i = 0, e = to.size(); i < e; ++i) {
				for (int j = 0, e = from.size(); j < e; ++j) {
					if (to[i] == from[j]) {
						out[i] = data_[j];
					}
				}
			}
			return out;
		}

		constexpr bool carry_sum_inc(int N)
		{
			for (int i = 0; i < size_; ++i) {
				if (++data_[i] < N) {
					return true; // no carry
				}
				data_[i] = 0; // reset and carry
			}
			return false; // overflow
		}
	};
}

template <>
struct std::formatter<ttl::ScalarIndex> {
	constexpr auto parse(format_parse_context& ctx)
	{
		return ctx.begin();
	}

	constexpr auto format(ttl::ScalarIndex const& index, auto& ctx)
	{
		auto out = ctx.out();
		for (auto&& i : index) {
			out = format_to(out, "{}", i);
		}
		return out;
	}
};
