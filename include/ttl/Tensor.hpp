#pragma once

#include "ttl/ParseTree.hpp"

namespace ttl
{
  template <class...>
  struct Tensor
  {
    std::string_view id_ = {};
    int rank_ = 0;
  };
}
