#pragma once

#include <type_traits>

namespace ttl
{
	template <typename T>
	concept is_equation = requires {
		typename std::remove_cvref_t<T>::is_equation_tag;
	};

	template <typename T>
	concept is_tree = requires {
		typename std::remove_cvref_t<T>::is_tree_tag;
	};

	template <typename T>
	concept is_system = requires {
		typename std::remove_cvref_t<T>::is_system_tag;
	};
}
