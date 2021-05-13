#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <optional>

namespace ttl
{
  struct TensorIndex
  {
    using tensor_index_tag = void;

    int size_ = 0;
    char data_[TTL_MAX_PARSE_INDEX] = {};

    constexpr TensorIndex() = default;

    constexpr TensorIndex(char c)
        : size_ { 1 }
        , data_ { c }
    {
    }

    constexpr TensorIndex(std::same_as<TensorIndex> auto const&... is)
    {
      ([&] {
        for (char c : is) {
          data_[size_++] = c;
        }
      }(), ...);
    }

    constexpr friend bool operator==(TensorIndex const& a, TensorIndex const& b)
    {
      return (a.size_ == b.size_) && std::equal(
        std::begin(a.data_), std::begin(a.data_) + a.size_,
        std::begin(b.data_), std::begin(b.data_) + b.size_);
    }

    constexpr friend auto operator<=>(TensorIndex const& a, TensorIndex const& b)
    {
      return std::lexicographical_compare_three_way(
        std::begin(a.data_), std::begin(a.data_) + a.size_,
        std::begin(b.data_), std::begin(b.data_) + b.size_);
    }

    constexpr auto size() const -> int
    {
      return size_;
    }

    constexpr auto rank() const -> int
    {
      return exclusive(*this).size();
    }

    constexpr auto to_string() const -> std::string_view
    {
      return { data_, data_ + size_ };
    }

    constexpr auto begin() const -> decltype(auto)
    {
      return std::begin(data_);
    }

    constexpr auto begin() -> decltype(auto)
    {
      return std::begin(data_);
    }

    constexpr auto end() const -> decltype(auto)
    {
      return begin() + size_;
    }

    constexpr auto end() -> decltype(auto)
    {
      return begin() + size_;
    }

    constexpr auto operator[](int i) const -> const char&
    {
      return data_[i];
    }

    constexpr auto operator[](int i) -> char&
    {
      return data_[i];
    }

    constexpr void push_back(char c)
    {
      data_[size_++] = c;
    }

    // Count the number of `c` in the index.
    constexpr auto count(char c) const -> int
    {
      int cnt = 0;
      for (char d : data_) {
        cnt += (c == d);
      }
      return cnt;
    }

    // Return the index of the first instance of `c` in the index, or nullopt.
    constexpr auto index_of(char c) const -> std::optional<int>
    {
      for (int i = 0, e = size_; i < e; ++i) {
        if (c == data_[i]) {
          return i;
        }
      }
      return std::nullopt;
    }

    // Hopefully obviously, search for chars in `search` and replace with the
    // corresponding char in `replace`.
    constexpr auto search_and_replace(TensorIndex const& search, TensorIndex const& replace)
      -> TensorIndex&
    {
      assert(search.size() == replace.size());
      for (char& c : data_) {
        if (auto&& i = search.index_of(c)) {
          c = replace[*i];
        }
      }
      return *this;
    }

    constexpr friend auto reverse(TensorIndex const& a)
      -> TensorIndex
    {
      TensorIndex out;
      for (int i = a.size_ - 1; i >= 0; --i) {
        out.push_back(a[i]);
      }
      return out;
    }

    constexpr friend auto unique(TensorIndex const& a)
      -> TensorIndex
    {
      TensorIndex out;
      for (char c : a) {
        if (out.count(c) == 0) {
          out.push_back(c);
        }
      }
      return out;
    }

    constexpr friend auto repeated(TensorIndex const& a)
      -> TensorIndex
    {
      TensorIndex out;
      for (char c : a) {
        if (a.count(c) > 1 && !out.index_of(c)) {
          out.push_back(c);
        }
      }
      return out;
    }

    constexpr friend auto exclusive(TensorIndex const& a)
      -> TensorIndex
    {
      TensorIndex out;
      for (char c : a) {
        if (a.count(c) == 1) {
          out.push_back(c);
        }
      }
      return out;
    }

    constexpr friend auto operator+=(TensorIndex& a, TensorIndex const& b)
      -> TensorIndex&
    {
      for (char c : b) a.push_back(c);
      return a;
    }

    constexpr friend auto operator+(TensorIndex const& a, TensorIndex const& b)
      -> TensorIndex
    {
      return { a, b };
    }

    constexpr friend auto operator&(TensorIndex const& a, TensorIndex const& b)
      -> TensorIndex
    {
      TensorIndex out;
      for (char c : a) {
        if (b.index_of(c)) {
          out.push_back(c);
        }
      }
      return out;
    }

    constexpr friend auto operator-(TensorIndex const& a, TensorIndex const& b)
      -> TensorIndex
    {
      TensorIndex out;
      for (char c : a) {
        if (!b.index_of(c)) {
          out.push_back(c);
        }
      }
      return out;
    }

    constexpr friend auto operator^(TensorIndex const& a, TensorIndex const& b)
      -> TensorIndex
    {
      return (a - b) + (b - a);
    }

    constexpr friend bool permutation(TensorIndex const& a, TensorIndex const& b)
    {
      return (a - b).size_ == 0 && (b - a).size_ == 0;
    }
  };

  template <class T>
  concept tensor_index_t = requires {
    typename std::remove_cvref_t<T>::tensor_index_tag;
  };

  using Index = TensorIndex;
}

#include <fmt/format.h>

template <>
struct fmt::formatter<ttl::TensorIndex>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.begin();
  }

  constexpr auto format(ttl::TensorIndex const& index, auto& ctx)
  {
    return format_to(ctx.out(), "{}", index.to_string());
  }
};
