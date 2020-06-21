#pragma once

#include <tuple>
#include <type_traits>

namespace ttl::mp {
template <typename>       constexpr inline bool is_tuple_v = false;
template <typename... Ts> constexpr inline bool is_tuple_v<std::tuple<Ts...>> = true;

template <typename T>
concept Tuple = is_tuple_v<std::remove_cvref_t<T>>;
}
