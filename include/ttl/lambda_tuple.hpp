#pragma once

#include <type_traits>
#include <utility>

// A tuple but implementation using a lambda function.
namespace ttl
{
  template <class T>
  struct lambda_tuple : T
  {
    using is_tuple_tag = void;

    constexpr lambda_tuple(T t)
        : T(std::move(t))
    {
    }
  };

  template <class T>
  concept is_tuple = requires {
    typename std::remove_cvref_t<T>::is_tuple_tag;
  };

  template <class... Ts>
  constexpr auto tuple(Ts&&... ts)
  {
    return lambda_tuple([... ts = std::forward<Ts>(ts)](auto f) -> decltype(auto) {
      return f(ts...);
    });
  };

  constexpr auto size(is_tuple auto&& t) -> std::size_t
  {
    return t([](auto&&...v) {
      return sizeof...(v);
    });
  }

  namespace lambda_tuple_get_detail
  {
    struct eat {
      constexpr eat(auto) {}
    };

    template <std::size_t... j>
    constexpr decltype(auto) inner(decltype(eat(j))..., auto&& m, ...) {
      return m;
    }

    template <std::size_t... j, class... Ts>
    constexpr decltype(auto) outer(std::index_sequence<j...>, Ts&&... v) {
      return inner<j...>(std::forward<Ts>(v)...);
    }
  };

  template <std::size_t N>
  constexpr auto get(is_tuple auto&& t) -> decltype(auto)
  {
    return t([]<class... Ts>(Ts&&... v) -> decltype(auto) {
        return lambda_tuple_get_detail::outer(std::make_index_sequence<N>(), std::forward<Ts>(v)...);
      });
  }
}
