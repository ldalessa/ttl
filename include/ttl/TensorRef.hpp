#pragma once

#include "ttl/TensorIndex.hpp"
#include <string_view>

namespace ttl
{
  struct TensorRef
  {
    using tensor_ref_tag = void;
    constexpr virtual auto   id() const -> std::string_view = 0;
    constexpr virtual auto rank() const -> int = 0;
    constexpr virtual auto outer_index() const -> TensorIndex = 0;
    constexpr virtual void print(fmt::memory_buffer& out) const = 0;
  };

  template <class T>
  concept tensor_ref = std::is_lvalue_reference_v<T> and requires {
    typename std::remove_cvref_t<T>::tensor_ref_tag;
  };

  template <class...>
  struct Tensor;
}
