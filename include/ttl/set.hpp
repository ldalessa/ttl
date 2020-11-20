#pragma once

#include <ce/dvector.hpp>
#include <optional>
#include <utility>

namespace ttl {
template <typename T>
struct set : ce::dvector<T> {
  using ce::dvector<T>::dvector;

  constexpr bool contains(const T& value) {
    for (auto&& t : *this) {
      if (t == value) return true;
    }
    return false;
  }

  template <typename... Ts>
  constexpr std::optional<int> find(Ts&&... ts) {
    T temp(std::forward<Ts>(ts)...);
    auto i = std::lower_bound(this->begin(), this->end(), temp);
    if (i == this->end()) return std::nullopt;
    if (*i != temp) return std::nullopt;
    return i - this->begin();
  }

  template <typename... Ts>
  constexpr void emplace(Ts&&... ts) {
    T temp(std::forward<Ts>(ts)...);
    if (!contains(temp)) {
      this->push_back(std::move(temp));
    }
//     // gcc produces different results in constexpr context for this code, I'm
//     // not sure why. I spent some time trying to reduce a testcase and couldn't
//     // get anything to fail.
//     //
//     const T& back = this->emplace_back(std::forward<Ts>(ts)...);
//     for (int i = 0; i < this->size() - 1; ++i) {
//       if (back == (*this)[i]) {
//         this->pop_back();
//         return;
//       }
//     }
  }
};
}
