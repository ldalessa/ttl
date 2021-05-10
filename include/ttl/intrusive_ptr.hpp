#pragma once

#include <cassert>
#include <cstdio>
#include <utility>

namespace ttl
{
  template <class T>
  struct intrusive_ptr
  {
    T *ptr_;

    constexpr ~intrusive_ptr()
    {
      dec();
    }

    constexpr intrusive_ptr(T *ptr)
        : ptr_(ptr)
    {
      inc();
    }

    constexpr intrusive_ptr(intrusive_ptr const& b)
        : ptr_(b.ptr_)
    {
      inc();
    }

    constexpr intrusive_ptr(intrusive_ptr&& b)
        : ptr_(std::exchange(b.ptr_, nullptr))
    {
    }

    constexpr intrusive_ptr& operator=(intrusive_ptr const& b) = delete;
    // {
    //   // Increment first prevents early delete, code works for self-loops.
    //   b.inc();
    //   dec();
    //   ptr_ = b.ptr_;
    //   return *this;
    // }

    constexpr intrusive_ptr& operator=(intrusive_ptr&& b) = delete;
    // {
    //   if (ptr_ != b.ptr_) {
    //     dec();
    //     ptr_ = std::exchange(b.ptr_, nullptr);
    //   }
    //   return *this;
    // }

    // constexpr intrusive_ptr& operator=(T* ptr)
    // {
    //   if (ptr) {
    //     ++ptr->count;
    //   }
    //   dec();
    //   ptr_ = ptr;
    //   return *this;
    // }

    // constexpr friend auto operator<=>(intrusive_ptr const&, intrusive_ptr const&) = default;

    constexpr auto operator->() const -> T*
    {
      return ptr_;
    }

    constexpr void inc() const
    {
      if (ptr_) {
        ++ptr_->count;
      }
    }

    constexpr void dec()
    {
      if (ptr_) {
        assert(ptr_->count > 0);
        if (--ptr_->count == 0) {
          std::exchange(ptr_, nullptr)->destroy();
        }
      }
    }
  };
}
