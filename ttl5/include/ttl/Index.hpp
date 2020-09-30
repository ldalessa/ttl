#pragma once

#include <cassert>
#include <concepts>
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

namespace ttl {
class alignas(int) Index {
  static constexpr int N = 8;
  int n_ = 0;
  char data_[N] = {};

 public:
  constexpr Index() = default;
  constexpr Index(char c) : n_(1), data_{c} {};
  constexpr Index(Index a, Index b) {
    for (char c : a) push(c);
    for (char c : b) push(c);
  }

  constexpr friend bool operator==(const Index& a, const Index& b) {
    if (a.n_ != b.n_) {
      return false;
    }
    for (int i = 0; i < a.n_; ++i) {
      if (a.data_[i] != b.data_[i]) {
        return false;
      }
    }
    return true;
  }

  friend std::ostream& operator<<(std::ostream& os, const Index& b) {
    return os << b.name();
  }

  constexpr int size() const {
    return n_;
  }

  constexpr const char& operator[](int i) const {
    assert(0 <= i && i < n_);
    return data_[i];
  }

  constexpr char& operator[](int i) {
    assert(0 <= i && i < n_);
    return data_[i];
  }

  constexpr const char* begin() const { return data_; }
  constexpr const char*   end() const { return data_ + n_; }

  constexpr       char* begin() { return data_; }
  constexpr const char*   end() { return data_ + n_; }

  constexpr auto rbegin() const { return std::reverse_iterator(data_ + n_); }
  constexpr auto   rend() const { return std::reverse_iterator(data_); }

  constexpr std::string_view name() const {
    return std::string_view(data_, n_);
  }

  constexpr void push(char c) {
    assert(n_ < N);
    data_[n_++] = c;
  }

  constexpr std::optional<int> find(char c) const {
    for (int i = 0; i < n_; ++i) {
      if (data_[i] == c) {
        return i;
      }
    }
    return std::nullopt;
  }

  constexpr int count(char c) const {
    int m = 0;
    for (int i = 0; i < n_; ++i) {
      m += (data_[i] == c);
    }
    return m;
  }

  constexpr Index& replace(const Index& is, const Index& with) {
    for (char& c : *this) {
      if (auto n = is.find(c)) {
        c = with[*n];
      }
    }
    return *this;
  }

  std::string str() const {
    return std::string(name());
  }
};

template <typename T>
concept IsIndex = std::same_as<std::remove_cvref_t<T>, Index>;

constexpr std::string_view name(const Index& i) {
  return i.name();
}

constexpr int size(Index i) {
  return i.size();
}

/// Reverse the Index string.
constexpr Index reverse(Index a) {
  Index out;
  for (auto i = std::rbegin(a), e = std::rend(a); i != e; ++i) {
    out.push(*i);
  }
  return out;
}

/// Return the unique characters in the Index.
constexpr Index unique(Index a) {
  Index out;
  for (char c : a) {
    if (!out.find(c)) {
      out.push(c);
    }
  }
  return out;
}

/// Return the characters that only appear once in the Index.
constexpr Index exclusive(Index a) {
  Index out;
  for (char c : a) {
    if (a.count(c) == 1) {
      out.push(c);
    }
  }
  return out;
}

/// Return the unique characters that appear more than once in the Index.
constexpr Index repeated(Index a) {
  Index out;
  for (char c : a) {
    if (a.count(c) > 1) {
      if (!out.find(c)) {
        out.push(c);
      }
    }
  }
  return out;
}

/// In-place set concatenation
constexpr Index& operator+=(Index& a, Index b) {
  for (char c : b) {
    a.push(c);
  }
  return a;
}

/// Set concatenation
constexpr Index operator+(Index a, Index b) {
  return a += b;
}

/// Set intersection
constexpr Index operator&(Index a, Index b) {
  Index out;
  for (char c : a) {
    if (b.find(c)) {
      out.push(c);
    }
  }
  return unique(out);
}

/// Set difference
constexpr Index operator-(Index a, Index b) {
  Index out;
  for (char c : a) {
    if (!b.find(c)) {
      out.push(c);
    }
  }
  return unique(out); // NB: is this really necessary?
}

/// Set symmetric difference
constexpr Index operator^(Index a, Index b) {
  return (a - b) + (b - a);
}

constexpr int order(Index i) {
  return exclusive(i).size();
}

constexpr bool permutation(Index a, Index b) {
  return size(a - b) == 0 && size(b - a) == 0;
}

constexpr Index replace(const Index& is, const Index& with, Index in) {
  return in.replace(is, with);
}
}
