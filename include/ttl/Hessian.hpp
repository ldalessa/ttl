#pragma once

#include "Index.hpp"
#include "Tensor.hpp"
#include <fmt/core.h>

namespace ttl {
struct Hessian
{
  Tensor tensor_;
  Index   inner_;

  constexpr Hessian(const Tensor& t) : tensor_(t)
  {
    assert(t.order() == 0);
  }

  constexpr Hessian(const Tensor& t, const Index& index)
      : tensor_(t)
      , inner_(index)
  {
    // anonymize the index
    Index search = unique(index);
    Index replace;
    for (int i = 0, e = search.size(); i < e; ++i) {
      replace.push_back(char('0' + i));
    }
    inner_.search_and_replace(search, replace);
  }

  constexpr friend bool operator==(const Hessian& a, const Hessian& b) {
    return a.tensor_ == b.tensor_ && a.inner_ == b.inner_;
  }

  constexpr friend bool operator<(const Hessian& a, const Hessian& b) {
    if (a.tensor_ < b.tensor_) return true;
    if (b.tensor_ < a.tensor_) return false;
    if (a.inner_ < b.inner_) return true;
    if (b.inner_ < a.inner_) return false;
    return false;
  }

  constexpr const Tensor& tensor() const {
    return tensor_;
  }

  constexpr Index inner() const {
    return inner_;
  }

  constexpr Index index() const {
    Index out;
    for (int i = 0, e = tensor_.order(); i < e; ++i) {
      out.push_back(inner_[i]);
    }
    return out;
  }

  constexpr Index partial() const {
    Index out;
    for (int i = tensor_.order(), e = inner_.size(); i < e; ++i) {
      out.push_back(inner_[i]);
    }
    return out;
  }

  constexpr Index outer() const {
    return unique(inner_);
  }

  constexpr int order() const {
    return outer().size();
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
    return format_to(ctx.out(), "{}({},{})", h.tensor(), h.index(), h.partial());
  }
};
