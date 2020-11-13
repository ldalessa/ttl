#pragma once

#include "Index.hpp"
#include "Node.hpp"
#include "Tensor.hpp"
#include <fmt/core.h>

namespace ttl {
struct Hessian {
  const Tensor* a_ = nullptr;
  Index dx_ = {};
  Index  i_ = {};

  constexpr Hessian() = default;

  constexpr Hessian(const Tensor* a)
      : a_(a)
  {
  }

  constexpr Hessian(const Tensor* a, Index all)
      : a_(a)
  {
    Index search = unique(all);
    Index replace;
    for (int i = 0, e = search.size(); i < e; ++i) {
      replace.push_back(char('0' + i));
    }
    all.search_and_replace(search, replace);

    int i = 0;
    for (int e = a->order(); i < e; ++i) {
      i_.push_back(all[i]);
    }
    for (int e = all.size(); i < e; ++i) {
      dx_.push_back(all[i]);
    }
  }

  constexpr friend bool operator==(const Hessian& a, const Hessian& b) {
    return a.a_ == b.a_ && a.i_ == b.i_ && a.dx_ == b.dx_;
  }

  constexpr friend bool operator<(const Hessian&a, const Hessian& b) {
    if (a.a_ < b.a_) return true;
    if (b.a_ < a.a_) return false;
    if (a.i_ < b.i_) return true;
    if (b.i_ < a.i_) return false;
    if (a.dx_ < b.dx_) return true;
    if (b.dx_ < a.dx_) return false;
    return false;
  }

  constexpr const Tensor* tensor() const {
    return a_;
  }

  constexpr Index index() const {
    return i_;
  }

  constexpr Index partial() const {
    return dx_;
  }

  constexpr Index inner() const {
    return index() + partial();
  }

  constexpr Index outer() const {
    return unique(inner());
  }

  constexpr int order() const {
    return outer().size();
  }

  constexpr friend int order(const Hessian& h) {
    return h.order();
  }
};
}

template <>
struct fmt::formatter<ttl::Hessian> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  constexpr auto format(const ttl::Hessian& h, FormatContext& ctx) {
    return format_to(ctx.out(), "{}({},{})", *h.tensor(), h.index(), h.partial());
  }
};

