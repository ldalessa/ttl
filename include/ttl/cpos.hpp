#pragma once

#include "ttl/Rational.hpp"
#include "ttl/TensorIndex.hpp"
#include "ttl/Tags.hpp"
#include <tag_invoke/tag_invoke.hpp>
#include <utility>

#ifndef FWD
#define FWD(a) std::forward<decltype(a)>(a)
#endif

namespace ttl
{
  inline constexpr struct size_tag_
  {
    constexpr friend auto tag_invoke(size_tag_, auto&& obj) -> int
      requires requires {FWD(obj).size();}
    {
      return FWD(obj).size();
    }

    constexpr friend auto tag_invoke(size_tag_, Rational const&) -> int
    {
      return 1;
    }

    constexpr friend auto tag_invoke(size_tag_, std::integral auto) -> int
    {
      return 1;
    }

    constexpr friend auto tag_invoke(size_tag_, std::floating_point auto) -> int
    {
      return 1;
    }

    constexpr auto operator()(auto&& obj) const -> int
    {
      return tag_invoke(*this, FWD(obj));
    }
  } size;

  inline constexpr struct outer_index_tag_
  {
    constexpr friend auto tag_invoke(outer_index_tag_, auto&& obj)
      -> TensorIndex
      requires requires { FWD(obj).outer_index(); }
    {
      return FWD(obj).outer_index();
    }

    constexpr friend auto tag_invoke(outer_index_tag_, Rational const&)
      -> TensorIndex
    {
      return {};
    }

    constexpr friend auto tag_invoke(outer_index_tag_, std::integral auto)
      -> TensorIndex
    {
      return {};
    }

    constexpr friend auto tag_invoke(outer_index_tag_, std::floating_point auto)
      -> TensorIndex
    {
      return {};
    }

    constexpr auto operator()(auto&& obj) const -> TensorIndex
    {
      return tag_invoke(*this, FWD(obj));
    }
  } outer_index;

  inline constexpr struct inner_index_tag_
  {
    constexpr friend auto tag_invoke(inner_index_tag_, auto&&) -> TensorIndex
    {
      return {};
    }

    constexpr auto operator()(auto&& obj) const -> TensorIndex
    {
      return tag_invoke(*this, FWD(obj));
    }
  } inner_index;

  inline constexpr struct rank_tag_
  {
    constexpr friend auto tag_invoke(rank_tag_, auto&& obj) -> int
    {
      if constexpr (requires { FWD(obj).rank(); }) {
        return FWD(obj).rank();
      }
      else {
        return size(outer_index(FWD(obj)));
      }
    }

    constexpr auto operator()(auto&& obj) const -> int
    {
      return tag_invoke(*this, FWD(obj));
    }
  } rank;

  inline constexpr struct tag_tag_
  {
    constexpr friend auto tag_invoke(tag_tag_, auto&& obj) -> TreeTag
      requires requires { FWD(obj).tag(); }
    {
      return FWD(obj).tag();
    }

    constexpr friend auto tag_invoke(tag_tag_, Rational const&) -> TreeTag
    {
      return LITERAL;
    }

    constexpr friend auto tag_invoke(tag_tag_, std::integral auto) -> TreeTag
    {
      return LITERAL;
    }

    constexpr friend auto tag_invoke(tag_tag_, std::floating_point auto) -> TreeTag
    {
      return LITERAL;
    }

    constexpr auto operator()(auto&& obj) const -> TreeTag
    {
      return tag_invoke(*this, FWD(obj));
    }
  } tag;

  inline constexpr struct to_string_tag_
  {
    template <class Obj>
    requires requires (Obj obj) { obj.to_string(); }
    constexpr friend auto tag_invoke(to_string_tag_, auto&& obj)
      -> decltype(FWD(obj).to_string())
    {
      return FWD(obj).tag();
    }

    constexpr auto operator()(auto&&... args) const
      noexcept(is_nothrow_tag_invocable_v<to_string_tag_, decltype(args)...>)
      -> tag_invoke_result_t<to_string_tag_, decltype(args)...>
    {
      return tag_invoke(*this, FWD(args)...);
    }
  } to_string;

  inline constexpr struct print_tag_
  {
    friend auto tag_invoke(print_tag_, auto&& obj, fmt::memory_buffer& out)
      requires requires {FWD(obj).print(out);}
    {
      return FWD(obj).print(out);
    }

    friend void tag_invoke(print_tag_ tag, auto&& obj, FILE *file)
    {
      fmt::memory_buffer out;
      tag_invoke(tag, FWD(obj), out);
      std::fwrite(out.data(), out.size(), 1, file);
    }

    void operator()(auto&& obj, auto&& out) const
      noexcept(is_nothrow_tag_invocable_v<print_tag_, decltype(obj), decltype(out)>)
    {
      return tag_invoke(*this, FWD(obj), FWD(out));
    }
  } print;

  inline constexpr struct dot_tag_
  {
    friend void tag_invoke(dot_tag_ tag, auto&& obj, FILE *file)
    {
      fmt::memory_buffer out;
      tag_invoke(tag, FWD(obj), out);
      std::fwrite(out.data(), out.size(), 1, file);
    }

    auto operator()(auto&& obj, auto&& buffer) const
      noexcept(is_nothrow_tag_invocable_v<dot_tag_, decltype(obj), decltype(buffer)>)
      -> tag_invoke_result_t<dot_tag_, decltype(obj), decltype(buffer)>
    {
      return tag_invoke(*this, FWD(obj), FWD(buffer));
    }
  } dot;

  inline constexpr struct visit_tag_
  {
    constexpr auto operator()(auto&&... args) const
      noexcept(is_nothrow_tag_invocable_v<visit_tag_, decltype(args)...>)
      -> tag_invoke_result_t<visit_tag_, decltype(args)...>
    {
      return tag_invoke(*this, FWD(args)...);
    }
  } visit;
}
