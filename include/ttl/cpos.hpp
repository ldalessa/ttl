#pragma once

#include "ttl/Rational.hpp"
#include "ttl/TensorIndex.hpp"
#include "ttl/Tags.hpp"
#include <tag_invoke/tag_invoke.hpp>
#include <string_view>
#include <utility>

#ifndef FWD
#define FWD(a) std::forward<decltype(a)>(a)
#endif

namespace ttl
{
  inline constexpr struct identifier_tag_
  {
    constexpr friend auto tag_invoke(identifier_tag_, auto&& obj) -> std::string_view
      requires requires {FWD(obj).id();}
    {
      return FWD(obj).id();
    }

    constexpr friend auto tag_invoke(identifier_tag_, Rational const&) -> std::string_view
    {
      return "";
    }

    constexpr friend auto tag_invoke(identifier_tag_, std::integral auto) -> std::string_view
    {
      return "";
    }

    constexpr friend auto tag_invoke(identifier_tag_, std::floating_point auto) -> std::string_view
    {
      return "";
    }

    constexpr auto operator()(auto&& obj) const -> std::string_view
    {
      return tag_invoke(*this, FWD(obj));
    }
  } identifier;

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
    friend auto tag_invoke(print_tag_, auto&& obj, fmt::memory_buffer& out, bool follow_links)
      requires requires {FWD(obj).print(out, follow_links);}
    {
      return FWD(obj).print(out, follow_links);
    }

    friend void tag_invoke(print_tag_ tag, auto&& obj, FILE *file, bool follow_links)
    {
      fmt::memory_buffer out;
      tag_invoke(tag, FWD(obj), out, follow_links);
      std::fwrite(out.data(), out.size(), 1, file);
    }

    void operator()(auto&& obj, auto&& out, bool follow_links = false) const
      noexcept(is_nothrow_tag_invocable_v<print_tag_, decltype(obj), decltype(out), bool>)
    {
      return tag_invoke(*this, FWD(obj), FWD(out), follow_links);
    }
  } print;

  inline constexpr struct dot_tag_
  {
    friend auto tag_invoke(dot_tag_, auto&& obj, fmt::memory_buffer& out, int i)
      requires requires {FWD(obj).dot(out, i);}
    {
      return FWD(obj).dot(out, i);
    }

    friend void tag_invoke(dot_tag_ tag, auto&& obj, FILE *file, int i)
    {
      fmt::memory_buffer out;
      fmt::format_to(out, "graph {} {{\n", identifier(obj));
      tag_invoke(tag, FWD(obj), out, 0);
      fmt::format_to(out, "}}\n");
      std::fwrite(out.data(), out.size(), 1, file);
    }

    void operator()(auto&& obj, auto&& buffer, int i = 0) const
      noexcept(is_nothrow_tag_invocable_v<dot_tag_, decltype(obj), decltype(buffer), int>)
    {
      return tag_invoke(*this, FWD(obj), FWD(buffer), i);
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
