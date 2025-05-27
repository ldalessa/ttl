#pragma once

#include <cstddef>
#include <ttl/parse/index_view.hpp>
#include <ttl/tensor.hpp>

namespace ttl::parse
{
	enum Tag {
		NONE,
		TENSOR,
		SUM,
		PRODUCT,
		DIFFERENCE,
		IDENTITY,
		NEGATE,
		DERIVATIVE,
		REBIND
	};

	struct Node {
		Tag _tag { NONE };
		std::size_t _left {};
		static constexpr std::size_t _right { 1 };
		union {
			struct {
			} _ {};

			struct {
				index_view _index;
			};

			struct {
				index_view _tensor_index;
				tensor _tensor_data;
			};
		};

		consteval Node() = default;

		consteval Node(Tag tag, std::size_t left, index_view index)
			: _tag(tag)
			, _left(left)
			, _index(index)
		{
		}

		consteval Node(tensor t, std::size_t left, index_view index)
			: _tag(TENSOR)
			, _left(left)
			, _tensor_index(index)
			, _tensor_data(t)
		{
		}

		constexpr auto tag() const -> Tag
		{
			return _tag;
		}

		constexpr auto index() const -> index_view
		{
			if (_tag == DERIVATIVE or _tag == REBIND) {
				return _index;
			}
			if (_tag == DERIVATIVE) {
				return _tensor_index;
			}
			expect(false);
		}

		constexpr auto tensor() const -> tensor
		{
			expect(_tag == TENSOR);
			return _tensor_data;
		}

		constexpr auto a() const -> Node const&
		{
			return *(this - _left);
		}

		constexpr auto b() const -> Node const&
		{
			return *(this - _right);
		}

		constexpr auto token() const -> char const*
		{
			static constexpr char const* tokens[] = { "[none]", "[tensor]", "+", "*", "-", "+", "-", "D" };
			return tokens[_tag];
		}
	};
}

template <>
struct std::formatter<ttl::parse::Node> {
	static constexpr auto parse(auto& ctx)
	{
		return ctx.begin();
	}

	static constexpr auto format(ttl::parse::Node const& node, auto& ctx)
	{
		switch (node._tag) {
		default:
			return ctx.out();
		case ttl::parse::TENSOR: {
			if (node._tensor_data.rank() == 0zu) {
				return std::format_to(ctx.out(), "{}", node.tensor());
			} else {
				return std::format_to(ctx.out(), "{}({})", node.tensor(), node.index());
			}
		}
		case ttl::parse::SUM:
		case ttl::parse::PRODUCT:
		case ttl::parse::DIFFERENCE: {
			auto out = std::format_to(ctx.out(), "({}", node.a());
			out = std::format_to(out, " {} ", node.token());
			return std::format_to(out, "{})", node.b());
		}
		case ttl::parse::IDENTITY:
		case ttl::parse::NEGATE:
		case ttl::parse::DERIVATIVE:
			return std::format_to(ctx.out(), "{}({},{})", node.token(), node.a(), node.index());

		case ttl::parse::REBIND:
			return std::format_to(ctx.out(), "{}({})", node.a(), node.index());
		}
	}
};
