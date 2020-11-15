#pragma once

#include "Index.hpp"
#include <ce/dvector.hpp>
#include <fmt/core.h>

namespace ttl {
struct ScalarIndex : ce::dvector<int> {
  using ce::dvector<int>::dvector;

  constexpr ScalarIndex
  select(const Index& from, const Index& to) const
  {
    assert(this->size() == from.size());
    ScalarIndex out;
    out.reserve(to.size());
    for (char c : to) {
      out.push_back((*this)[*from.index_of(c)]);
    }
    return out;
  }
};
}

template <>
struct fmt::formatter<ttl::ScalarIndex>
{
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::ScalarIndex& index, FormatContext& ctx)
  {
    auto out = ctx.out();
    for (auto&& i : index) {
      out = format_to(out, "{}", i);
    }
    return out;
  }
};
