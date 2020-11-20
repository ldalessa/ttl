#pragma once

#include <eve/wide.hpp>
#include <cstdlib>
#include <limits>

namespace ttl
{
template <class T>
struct SIMDAllocator
{
  typedef T value_type;

  [[nodiscard]] T* allocate(std::size_t n) {
    static constexpr auto align = eve::wide<T>::static_alignment;
    return static_cast<T*>(std::aligned_alloc(align, n * sizeof(T)));
  }

  void deallocate(T* p, std::size_t n) noexcept {
    std::free(p);
  }
};
}
