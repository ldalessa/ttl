#pragma once

namespace ttl::concepts
{
	template <class T>
	concept index = requires {
		typename T::_ttl_index_tag;
	};
}