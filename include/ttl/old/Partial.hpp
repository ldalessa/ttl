#pragma once

#include "Tensor.hpp"
#include "utils.hpp"
#include <algorithm>
#include <string>
#include <string_view>


namespace ttl {
struct Partial {
  Tensor tensor = {};
  int component = 0;
  int        dx[8] = {};
  int         N = 0;

  constexpr Partial() = default;

  constexpr Partial(int N, const Tensor& t, auto const& index)
      : tensor(t)
      , N(N)
  {
    assert(N <= std::size(dx));
    assert(t.order() <= std::ssize(index));

    int i = 0;
    for (int e = t.order(); i < e; ++i) {
      component += utils::pow(N, i) * index[i];
    }
    for (int e = index.size(); i < e; ++i) {
      ++dx[index[i]];
    }
  }

  constexpr bool operator==(const Partial& rhs) {
    if (tensor != rhs.tensor) return false;
    if (component != rhs.component) return false;
    for (int n = 0; n < N; ++n) {
      if (dx[n] != rhs.dx[n]) return false;
    }
    return true;
  }

  constexpr bool operator<(const Partial& rhs) const {
    if (partial_mask() < rhs.partial_mask()) return true;
    if (rhs.partial_mask() < partial_mask()) return false;
    for (int n = 0; n < N; ++n) {
      if (dx[n] < rhs.dx[n]) return true;
      if (rhs.dx[n] < dx[n]) return false;
    }
    if (id() < rhs.id()) return true;
    if (rhs.id() < id()) return false;
    if (component < rhs.component) return true;
    if (rhs.component < component) return false;
    return false;
  }

  constexpr std::string_view id() const {
    return tensor.id();
  }

  constexpr int partial_mask() const {
    int out = 0;
    for (int n = 0; n < N; ++n) {
      if (dx[n]) {
        out += utils::pow(2, n);
      }
    }
    return out;
  }

  std::string partial_string() const {
    constexpr const char ids[] = "xyz";
    std::string str;
    for (int n = 0; n < N; ++n) {
      for (int i = 0; i < dx[n]; ++i) {
        str += ids[n];
      }
    }
    return str;
  }
};

template <int M>
struct PartialManifest {
  Partial data[M];
  int N;

  constexpr PartialManifest(int N, utils::set<Partial>&& partials)
      : N(N)
  {
    assert(M == partials.size());
    for (int i = 0; auto&& p : partials) {
      assert(N == p.N);
      data[i++] = p;
    }
    std::sort(data, data + M);
  }

  constexpr static int      size()       { return M; }
  constexpr decltype(auto) begin() const { return data + 0; }
  constexpr decltype(auto)   end() const { return data + M; }

  constexpr decltype(auto) operator[](int i) const { assert(0 <= i && i < M);
    return data[i];
  }

  constexpr decltype(auto) operator[](int i) { assert(0 <= i && i < M);
    return data[i];
  }

  constexpr auto dx(int mask) const {
    auto a = std::ranges::lower_bound(data, mask, std::less{}, [](const Partial& dx) {
      return dx.partial_mask();
    });
    auto b = std::ranges::lower_bound(data, mask + 1, std::less{}, [](const Partial& dx) {
      return dx.partial_mask();
    });

    struct IndexRange {
      int i;
      int j;

      constexpr IndexRange(int i, int j) : i(i), j(j) {}

      struct iterator {
        int i;
        constexpr decltype(auto) operator*()  const { return i; }
        constexpr decltype(auto) operator++() { return (++i, *this); }
        constexpr bool operator==(const iterator& b) const { return i == b.i; }
        constexpr auto operator<=>(const iterator& b) const { return i <=> b.i; }
      };

      constexpr int       size() const { return j - i; };
      constexpr iterator begin() const { return { i }; }
      constexpr iterator   end() const { return { j }; }
    };

    return IndexRange(a - data, b - data);
  }

  constexpr auto fields() const {
    return dx(0);
  }

  constexpr std::optional<int>
  find(const Tensor& t, const ce::dvector<int>& index) const
  {
    Partial p(N, t, index);
    auto begin = data;
    auto   end = data + M;
    if (auto i = std::lower_bound(begin, end, p); i < end) {
      if (*i == p) {
        return i - begin;
      }
    }
    return std::nullopt;
  }
};
}

template <>
struct fmt::formatter<ttl::Partial> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const ttl::Partial& p, FormatContext& ctx) {
    return format_to(ctx.out(), "{} {} d{}", *p.tensor, p.component,
                     p.partial_string());
  }
};
