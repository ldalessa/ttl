#pragma once

import std;

namespace ttl::concepts
{
	template <class T>
	concept character = std::same_as<T, char> or std::same_as<T, char8_t> or std::same_as<T, char16_t> or std::same_as<T, char32_t>;
}