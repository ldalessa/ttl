#pragma once

#include <utility>

// A chuple is like a tuple but implementation using a lambda function.
//
// The `ch` comes from Alonzo Church.

namespace ttl {
template <typename T>
struct chuple_t : T {
  constexpr static std::true_type is_chuple = {};
  constexpr chuple_t(T&& t) : T(std::move(t)) {}
};

template <typename T>
concept is_chuple = requires (T t) { {t.is_chuple}; };

template<typename...T>
constexpr auto chuple(T&&...v) {
  return chuple_t([v...](auto f) -> decltype(auto) {
    return f(v...);
  });
};

constexpr auto size(is_chuple auto&& t) {
  return t([](auto&&...v) {
    return sizeof...(v);
  });
}

namespace chuple_get_detail {
struct eat {
  constexpr eat(auto) {}
};

template <std::size_t... j>
constexpr decltype(auto) inner(decltype(eat(j))..., auto&& m, ...) {
  return m;
}

template <std::size_t... j, typename... Ts>
constexpr decltype(auto) outer(std::index_sequence<j...>, Ts&&... v) {
  return inner<j...>(std::forward<Ts>(v)...);
}
};

template <int i>
constexpr auto& get(is_chuple auto&& t)  {
  return t([]<typename... Ts>(Ts&&... v) -> auto& {
      return chuple_get_detail::outer(std::make_index_sequence<i>(), std::forward<Ts>(v)...);
    });
}
}
