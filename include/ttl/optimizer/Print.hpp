#pragma once

#include "ttl/optimizer/Nodes.hpp"
#include <fmt/format.h>
#include <string_view>

namespace ttl::optimizer
{
  using namespace std::literals;

  struct Print
  {
    fmt::memory_buffer out;

    void format(const auto&... args)
    {
      fmt::format_to(out, args...);
    }

    void write(FILE* file)
    {
      std::fwrite(out.data(), out.size(), 1, file);
    }

    auto operator()(node_ptr node) -> auto&
    {
      node.visit(*this);
      return out;
    }

    void operator()(Binary* node)
    {
      out.append("("sv);
      node->a.visit(*this);
      fmt::format_to(out, " {} ", node->tag);
      node->b.visit(*this);
      out.append(")"sv);
    }

    void operator()(Pow* node)
    {
      out.append("("sv);
      node->a.visit(*this);
      fmt::format_to(out, "){}", node->tag);
      node->b.visit(*this);
    }

    void operator()(Unary* node)
    {
      fmt::format_to(out, "{}(", node->tag);
      node->a.visit(*this);
      out.append(")"sv);
    }

    void operator()(Bind* bind)
    {
      fmt::format_to(out, "{}(", bind->tag);
      bind->a.visit(*this);
      fmt::format_to(out, ",{})", bind->index);
    }

    void operator()(Partial* partial)
    {
      fmt::format_to(out, "{}(", partial->tag);
      partial->a.visit(*this);
      fmt::format_to(out, ",{})", partial->index);
    }

    void operator()(Literal* lit)
    {
      if (lit->tag == DOUBLE) {
        fmt::format_to(out, "{}", lit->d * as<double>(lit->q));
      }
      else {
        fmt::format_to(out, "{}", lit->q);
      }
    }

    void operator()(Tensor* tensor)
    {
      if (tensor->index.size()) {
        fmt::format_to(out, "{}({})", *tensor->tensor, tensor->index);
      }
      else {
        fmt::format_to(out, "{}", *tensor->tensor);
      }
    }

    void operator()(Scalar* scalar)
    {
      if (scalar->index.size()) {
        fmt::format_to(out, "{}({})", *scalar->tensor, scalar->index);
      }
      else {
        fmt::format_to(out, "{}", *scalar->tensor);
      }
    }

    void operator()(Delta* δ)
    {
      fmt::format_to(out, "{}({})", δ->tag, δ->index);
    }

    void operator()(Epsilon* ε)
    {
      fmt::format_to(out, "{}({})", ε->tag, ε->index);
    }
  };
}
