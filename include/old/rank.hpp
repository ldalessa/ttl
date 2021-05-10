#pragma once

#include "tag_invoke/tag_invoke.hpp"
#include <concepts>

namespace ttl
{
  struct rank_tag
  {
    constexpr auto operator()(auto&&... args) const
      noexcept(is_nothrow_tag_invocable_v<rank_tag, decltype(args)...>)
      -> tag_invoke_result_t<rank_tag, decltype(args)...>
    {
      return tag_invoke(*this, std::forward<decltype(args)>(args)...);
    }
  };

  constexpr inline rank_tag rank = {};

  constexpr static auto tag_invoke(rank_tag, std::integral auto) -> int
  {
    return 0;
  }

  constexpr static auto tag_invoke(rank_tag, std::floating_point auto) -> int
  {
    return 0;
  }
}
