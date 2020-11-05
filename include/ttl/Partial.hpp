#pragma once

#include "Hessian.hpp"
#include "Tensor.hpp"
#include "Index.hpp"
#include "utils.hpp"
#include <ranges>

namespace ttl {
template <int N> requires(N > 0)
struct Partial {
  Tensor tensor;
  int component = 0;
  int     dx[N] = {};

  constexpr Partial(const Hessian& h, int index[])
      : tensor(h.tensor())
  {
    Index outer = h.outer();

    for (int i = 0; auto&& c : h.index()) {
      component += utils::pow(N, i++) * index[*utils::index_of(outer, c)];
    }

    for (auto&& c : h.partial()) {
      ++dx[index[*utils::index_of(outer, c)]];
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

template <int N, int M>
struct PartialManifest {
  Partial<N> data[M];

  template <typename... Ts>
  requires(std::same_as<std::remove_cvref_t<Ts>, Partial<N>> && ...)
  constexpr PartialManifest(Ts&&... ps)
      : data { std::forward<Ts>(ps)... }
  {
  }

  constexpr static int      size()       { return M; }
  constexpr decltype(auto) begin() const { return data + 0; }
  constexpr decltype(auto)   end() const { return data + size(); }

  constexpr decltype(auto) operator[](int i) const { assert(0 <= i && i < M);
    return data[i];
  }

  constexpr decltype(auto) operator[](int i) { assert(0 <= i && i < M);
    return data[i];
  }

  constexpr auto dx(int mask) const {
    auto a = std::ranges::lower_bound(data, mask, std::less{}, [](const Partial<N>& dx) {
      return dx.partial_mask();
    });
    auto b = std::ranges::lower_bound(data, mask + 1, std::less{}, [](const Partial<N>& dx) {
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
};

template <int N>
PartialManifest(Partial<N>, std::same_as<Partial<N>> auto... rest) ->
  PartialManifest<N, 1 + sizeof...(rest)>;
}

template <int N>
struct fmt::formatter<ttl::Partial<N>> {
  constexpr auto parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const ttl::Partial<N>& p, FormatContext& ctx) {
    return format_to(ctx.out(), "{} {} d{}", p.tensor, p.component,
                     p.partial_string());
  }
};
