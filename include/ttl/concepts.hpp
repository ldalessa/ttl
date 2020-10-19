#pragma once

#include <concepts>

namespace ttl {
template <typename T>
concept is_equation = requires(T t) { t.is_equation_tag; };

template <typename T>
concept is_tree = requires (T t) { t.is_tree_tag; };
}
