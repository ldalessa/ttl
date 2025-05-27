#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <format>
#include <iterator>
#include <ttl/concepts/character.hpp>
#include <ttl/utils/count_output.hpp>
#include <ttl/utils/cstring_sentinel.hpp>

namespace ttl
{
	template <std::size_t N = 1, concepts::character CharT = char, CharT _projected = CharT { '?' }>
	struct index {
		static_assert(N > 0);

		using _ttl_index_tag = void;

		CharT _data[N] {};

		consteval index() = default;

		consteval index(CharT const c)
			: _data { c, CharT() }
		{
		}

		consteval index(CharT const (&str)[N])
		{
			std::ranges::copy_n(str, N, _data);
			assert(_validate() == N);
		}

		template <std::size_t S, std::size_t T>
		constexpr index(index<S, CharT> const& a, index<T, CharT> const& b)
		{
			static_assert(N == S + T - 1zu);
			std::ranges::copy(a, _data);
			std::ranges::copy(b, _data + S - 1zu);
			assert(_validate() == N);
		}

		static consteval auto size() -> std::size_t
		{
			return N;
		}

		constexpr auto begin(this auto&& self) -> decltype(auto)
		{
			return FWD(self)._data + 0zu;
		}

		static constexpr auto end()
		{
			return cstring_sentinel;
		}

		friend constexpr auto operator==(index const& a, index const& b) -> bool
		{
			return std::ranges::equal(a, b);
		}

		friend constexpr auto operator<=>(index const& a, index const& b)
		{
			return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end());
		}

		template <std::size_t M>
		friend constexpr auto operator+(index const& a, index<M> const& b)
		{
			return index<N + M - 1zu>(a, b);
		}

		/// Copy the characters that occur once into the output.
		///
		/// This does not copy the null terminator.
		constexpr auto outer(std::output_iterator<CharT> auto out) const
		{
			return std::ranges::copy_if(_data, cstring_sentinel, out, [&](CharT const& c) {
				return c != _projected and std::ranges::count(_data, c) == 1zu;
			}).out;
		}

		/// Copy the characters that occur twice into the output.
		constexpr auto contracted(std::output_iterator<CharT> auto out) const
		{
			return std::ranges::copy_if(_data, cstring_sentinel, out, [&](CharT const& c) {
				return c != _projected and std::ranges::count(_data, &c, c) == 1zu;
			}).out;
		}

		/// Copy the projected indices.
		constexpr auto projected(std::output_iterator<CharT> auto out) const
		{
			return std::ranges::copy_if(_data, cstring_sentinel, out, [&](CharT const& c) {
				return c == _projected;
			}).out;
		}

		/// Concatenate the outer and contracted indices.
		///
		/// This does not copy the null terminator.
		constexpr auto inner(std::output_iterator<CharT> auto out) const
		{
			return projected(contracted(outer(out)));
		}

		constexpr auto outer_size() const -> std::size_t
		{
			return outer(count_output) + 1zu;
		}

		constexpr auto contracted_size() const -> std::size_t
		{
			return contracted(count_output) + 1zu;
		}

		constexpr auto projected_size() const -> std::size_t
		{
			return std::ranges::count(_data, _projected) + 1zu;
		}

		constexpr auto inner_size() const -> std::size_t
		{
			return inner(count_output) + 1zu;
		}

		constexpr auto order() const -> std::size_t
		{
			return outer_size() - 1zu;
		}

	private:
		constexpr auto _validate() const -> std::size_t
		{
			for (auto const& c : _data) {
				if (c == _projected) {
					continue;
				}
				if (std::ranges::count(_data, &c, c) > 1) {
					return &c - _data;
				};
			}
			return N;
		}
	};

	template <concepts::character CharT>
	index(CharT const) -> index<2>;

	template <std::size_t S, std::size_t T>
	index(index<S>, index<T>) -> index<S + T - 1zu>;

	template <index i>
	inline constexpr auto outer = [] {
		index<i.outer_size()> out {};
		i.outer(out.begin());
		return out;
	}();

	template <index i>
	inline constexpr auto contracted = [] {
		index<i.contracted_size()> out {};
		i.contracted(out.begin());
		return out;
	}();

	template <index i>
	inline constexpr auto projected = [] {
		index<i.projected_size()> out {};
		i.projected(out.begin());
		return out;
	}();

	template <index i>
	inline constexpr auto inner = [] {
		index<i.inner_size()> out {};
		i.inner(out.begin());
		return out;
	}();

}

template <std::size_t N, class CharT>
struct std::formatter<ttl::index<N, CharT>> : std::formatter<CharT const*> {
	constexpr auto parse(auto& ctx)
	{
		return std::formatter<CharT const*>::parse(ctx);
	}

	constexpr auto format(ttl::index<N, CharT> const& index, auto& ctx) const
	{
		return std::formatter<CharT const*>::format(index._data, ctx);
	}
};