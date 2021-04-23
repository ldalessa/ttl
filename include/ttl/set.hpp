#pragma once

#include <ce/dvector.hpp>
#include <optional>
#include <utility>

namespace ttl
{
  template <typename T>
  struct set : ce::dvector<T>
  {
    using ce::dvector<T>::dvector;

    constexpr bool contains(const T& value) const
    {
      for (auto&& t : *this) {
        if (t == value) return true;
      }
      return false;
    }

    constexpr auto find(T const& t) const -> std::optional<int>
    {
      // set's aren't sorted
      for (auto i = this->begin(), e = this->end(); i != e; ++i) {
        if (*i == t) {
          return std::distance(this->begin(), i);
        }
      }
      return std::nullopt;
    }

    template <typename... Ts>
    constexpr auto find(Ts&&... ts) const -> std::optional<int>
    {
      return find(T(std::forward<Ts>(ts)...));
    }

    template <typename... Ts>
    constexpr bool emplace(Ts&&... ts)
    {
      T temp(std::forward<Ts>(ts)...);
      if (!contains(temp)) {
        this->push_back(std::move(temp));
        return true;
      }
      return false;
    }

    constexpr auto sort() -> set&
    {
      std::sort(this->begin(), this->end());
      return *this;
    }

    template <std::size_t M>
    constexpr friend auto to_array(set const& self) -> std::array<T, M>
    {
      assert(M == self.size());
      std::array<T, M> out;
      std::copy_n(self.begin(), M, out.begin());
      return out;
    }
  };
}
