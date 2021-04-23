#pragma once

#include <concepts>

namespace ttl {
  template <class T>
  concept is_equation = requires {
    typename std::remove_cvref_t<T>::is_equation_tag;
  };

  template <class T>
  concept is_parse_tree = requires {
    typename std::remove_cvref_t<T>::is_parse_tree_tag;
  };

  template <class T>
  concept is_index = requires {
    typename std::remove_cvref_t<T>::is_index_tag;
  };

  template <class T>
  concept is_system = requires {
    typename std::remove_cvref_t<T>::is_system_tag;
  };
}
