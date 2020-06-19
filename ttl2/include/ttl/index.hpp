#pragma once

#include <cassert>
#include <iterator>
#include <optional>
#include <string_view>

namespace ttl {
class index {
  static constexpr int N = 8;
  char data_[N] = {};
  int n_ = 0;

 public:
  constexpr index() = default;
  constexpr index(char c) : data_{c}, n_(1) {};
  constexpr index(index a, index b) {
    for (char c : a) push(c);
    for (char c : b) push(c);
  }

  friend constexpr bool operator==(const index& a, const index& b) {
    if (a.n_ != b.n_) return false;
    for (int i = 0; i < a.n_; ++i) {
      if (a.data_[i] != b.data_[i]) return false;
    }
    return true;
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

  constexpr const char* begin() const {
    return data_;
  }
  constexpr auto rbegin() const {
    return std::reverse_iterator(data_ + n_);
  }

  constexpr const char* end() const {
    return data_ + n_;
  }

  constexpr auto rend() const {
    return std::reverse_iterator(data_);
  }

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
};

template <typename T>
concept Index = std::same_as<std::remove_cvref_t<T>, index>;

template <char Id>
constexpr inline index idx = { Id };

constexpr std::string_view name(const index& i) {
  return i.name();
}

constexpr int size(index i) {
  return i.size();
}

/// Reverse the index string.
constexpr index reverse(index a) {
  index out;
  for (auto i = std::rbegin(a), e = std::rend(a); i != e; ++i) {
    out.push(*i);
  }
  return out;
}

/// Return the unique characters in the index.
constexpr index unique(index a) {
  index out;
  for (char c : a) {
    if (!out.find(c)) {
      out.push(c);
    }
  }
  return out;
}

/// Return the characters that only appear once in the index.
constexpr index exclusive(index a) {
  index out;
  for (char c : a) {
    if (a.count(c) == 1) {
      out.push(c);
    }
  }
  return out;
}

/// Return the unique characters that appear more than once in the index.
constexpr index repeated(index a) {
  index out;
  for (char c : a) {
    if (a.count(c) > 1) {
      if (!out.find(c)) {
        out.push(c);
      }
    }
  }
  return out;
}

/// Set concatenation
constexpr index operator+(index a, index b) {
  return index(a, b);
}

/// Set intersection
constexpr index operator&(index a, index b) {
  index out;
  for (char c : a) {
    if (b.find(c)) {
      out.push(c);
    }
  }
  return unique(out);
}

/// Set difference
constexpr index operator-(index a, index b) {
  index out;
  for (char c : a) {
    if (!b.find(c)) {
      out.push(c);
    }
  }
  return unique(out);
}

/// Set symmetric difference
constexpr index operator^(index a, index b) {
  return (a - b) + (b - a);
}

constexpr int order(index i) {
  return exclusive(i).size();
}

constexpr bool permutation(index a, index b) {
  return size(a - b) == 0 && size(b - a) == 0;
}
}
