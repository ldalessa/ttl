#pragma once

#include <algorithm>
#include <cstddef>
#include <format>
#include <ttl/parse/Node.hpp>
#include <ttl/parse/index_view.hpp>
#include <ttl/tensor.hpp>

namespace ttl::parse
{
	template <std::size_t _size, std::size_t _index_length>
	struct Tree {
		static_assert(_size > 0);

		Node _nodes[_size];
		char _indices[_index_length];

		consteval Tree() = default;

		consteval Tree(tensor const& t, auto... is)
		{
			static_assert(sizeof...(is) == _index_length);
			char const index[] { (is.c)... };
			std::ranges::copy(index, _indices);
			_nodes[0] = Node(t, 0, index_view(_indices, _index_length));
		}

		template <std::size_t A, std::size_t N, std::size_t B, std::size_t M>
		consteval Tree(Tag tag, Tree<A, N> const& a, Tree<B, M> const& b)
		{
			static_assert(A + B + 1 == _size);
			static_assert(N + M == _index_length);
			std::ranges::copy(a._nodes, _nodes + 0);
			std::ranges::copy(b._nodes, _nodes + A);
			_nodes[A + B]._tag = tag;
			_nodes[A + B]._left = A;

			std::ranges::copy(a._indices, _indices + 0);
			std::ranges::copy(b._indices, _indices + N);
		}

		template <std::size_t A, std::size_t N>
		consteval Tree(Tree<A, N> const& a, Tag tag, auto... is)
		{
			static_assert(A + 2 == _size);
			static_assert(N + sizeof...(is) == _index_length);

			char const index[] { (is.c)... };
			std::ranges::copy(a._indices, _indices + 0);
			std::ranges::copy(index, _indices + N);
			std::ranges::copy(a._nodes, _nodes);

			_nodes[A + 0] = Node();
			_nodes[A + 1] = Node(tag, 2, index_view(_indices + N, _indices + _index_length));
		}

		constexpr auto top() const -> Node const&
		{
			return _nodes[_size - 1];
		}

		constexpr auto operator()(auto... is) const -> Tree<_size + 2, _index_length + sizeof...(is)>
		{
			return { *this, REBIND, index(is)... };
		}

		template <std::size_t B, std::size_t M>
		friend consteval auto operator+(Tree const& a, Tree<B, M> const& b) -> Tree<_size + B + 1, _index_length + M>
		{
			return { SUM, a, b };
		}

		template <std::size_t B, std::size_t M>
		friend consteval auto operator*(Tree const& a, Tree<B, M> const& b) -> Tree<_size + B + 1, _index_length + M>
		{
			return { PRODUCT, a, b };
		}

		template <std::size_t B, std::size_t M>
		friend consteval auto operator-(Tree const& a, Tree<B, M> const& b) -> Tree<_size + B + 1, _index_length + M>
		{
			return { DIFFERENCE, a, b };
		}

		friend consteval auto operator+(Tree const& a) -> Tree<_size + 2, _index_length>
		{
			return { IDENTITY, a, Tree<1, 0>() };
		}

		friend consteval auto operator-(Tree const& a) -> Tree<_size + 2, _index_length>
		{
			return { NEGATE, a, Tree<1, 0>() };
		}

		friend consteval auto D(Tree const& a, auto... is) -> Tree<_size + 2, _index_length + sizeof...(is)>
		{
			return { a, DERIVATIVE, ttl::index(is)... };
		}
	};
}

namespace ttl
{
	consteval auto tensor::_bind(auto... is) const -> parse::Tree<1, sizeof...(is)>
	{
		return { *this, is... };
	}
}

template <std::size_t A, std::size_t N>
struct std::formatter<ttl::parse::Tree<A, N>> {
	static constexpr auto parse(auto& ctx)
	{
		return ctx.begin();
	}

	static constexpr auto format(ttl::parse::Tree<A, N> const& a, auto& ctx)
	{
		return std::format_to(ctx.out(), "{}", a.top());
	}
};