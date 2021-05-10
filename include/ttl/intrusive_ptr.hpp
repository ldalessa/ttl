#pragma once

#include <cassert>
#include <cstdio>
#include <utility>

namespace ttl
{
  template <class T>
  struct unique_ptr
  {
    T *ptr_ = nullptr;

    constexpr ~unique_ptr()
    {
      if constexpr (requires {ptr_->destroy();}) {
        ptr_->destroy();
      }
      else {
        delete ptr_;
      }
    }

    constexpr unique_ptr() = default;

    constexpr unique_ptr(T *ptr)
        : ptr_(ptr)
    {
    }

    constexpr unique_ptr(unique_ptr const& b) = delete;

    constexpr unique_ptr(unique_ptr&& b)
        : ptr_(std::exchange(b.ptr_, nullptr))
    {
    }

    constexpr unique_ptr& operator=(unique_ptr const& b) = delete;

    constexpr unique_ptr& operator=(unique_ptr&& b)
    {
      ptr_ = std::exchange(b.ptr_, nullptr);
    }

    constexpr auto operator*() const -> T&
    {
      return *ptr_;
    }

    constexpr auto operator->() const -> T*
    {
      return ptr_;
    }
  };

  template <class T>
  struct intrusive_ptr
  {
    T *ptr_ = nullptr;

    constexpr ~intrusive_ptr()
    {
      dec();
    }

    constexpr intrusive_ptr() = default;

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

    constexpr intrusive_ptr& operator=(intrusive_ptr const& b)
    {
      // Increment first prevents early delete, code works for self-loops.
      b.inc();
      dec();
      ptr_ = b.ptr_;
      return *this;
    }

    constexpr intrusive_ptr& operator=(intrusive_ptr&& b)
    {
      if (ptr_ != b.ptr_) {
        dec();
        ptr_ = std::exchange(b.ptr_, nullptr);
      }
      return *this;
    }

    constexpr intrusive_ptr& operator=(T* ptr)
    {
      if (ptr) {
        ++ptr->count;
      }
      dec();
      ptr_ = ptr;
      return *this;
    }

    constexpr explicit operator bool() const
    {
      return ptr_ != nullptr;
    }

    constexpr friend void swap(intrusive_ptr& a, intrusive_ptr& b) {
      std::swap(a.ptr_, b.ptr_);
    }

    constexpr friend auto operator<=>(intrusive_ptr const&, intrusive_ptr const&) = default;

    constexpr auto operator*() const -> T&
    {
      return *ptr_;
    }

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
          if constexpr (requires {ptr_->destroy();}) {
            std::exchange(ptr_, nullptr)->destroy();
          }
          else {
            delete std::exchange(ptr_, nullptr);
          }
        }
      }
    }
  };
}
