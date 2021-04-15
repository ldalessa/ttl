#pragma once

#include <utility>

namespace ttl
{
  template <class T>
  struct intrusive_ptr
  {
    T* ptr_ = nullptr;

    constexpr ~intrusive_ptr()
    {
      dec_();
    }

    constexpr intrusive_ptr() = default;

    constexpr intrusive_ptr(T* ptr)
        : ptr_(ptr)
    {
      inc_();
    }

    template <class... Ts> requires (std::constructible_from<T, Ts...>)
    constexpr intrusive_ptr(Ts&&... ts)
      : ptr_(new T(std::forward<Ts>(ts)...))
    {
      inc_();
    }

    constexpr intrusive_ptr(intrusive_ptr const& b)
        : ptr_(b.ptr_)
    {
      inc_();
    }

    constexpr intrusive_ptr(intrusive_ptr&& b)
        : ptr_(std::exchange(b.ptr_, nullptr))
    {
    }

    constexpr auto operator=(intrusive_ptr const& b) -> decltype(auto)
    {
      dec_();
      ptr_ = b.ptr_;
      inc_();
      return *this;
    }

    constexpr auto operator=(intrusive_ptr&& b) -> decltype(auto)
    {
      dec_();
      ptr_ = std::exchange(b.ptr_, nullptr);
      return *this;
    }

    constexpr friend bool operator==(intrusive_ptr const& a, std::nullptr_t) {
      return a.ptr_ == nullptr;
    }

    constexpr auto operator->() -> T*
    {
      return ptr_;
    }

    constexpr auto operator->() const -> T const*
    {
      return ptr_;
    }

    constexpr void inc_()
    {
      if (ptr_) {
        if (++ptr_->count_ <= 0) {
          assert(false);
        }
      }
    }

    constexpr void dec_()
    {
      if (ptr_) {
        if (--ptr_->count_ == 0)
        {
          delete ptr_;
          ptr_ = nullptr;
        }
      }
    }
  };
}
