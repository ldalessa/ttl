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

  constexpr friend bool operator==(const ScalarIndex& a, const ScalarIndex& b) {
    if (a.size() != b.size()) return false;
    for (int i = 0, e = a.size(); i < e; ++i) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  constexpr friend bool operator<(const ScalarIndex& a, const ScalarIndex& b) {
    if (a.size() < b.size()) return true;
    if (b.size() < a.size()) return false;
    for (int i = 0, e = a.size(); i < e; ++i) {
      if (a[i] < b[i]) return true;
      if (b[i] < a[i]) return false;
    }
    return false;
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
