#pragma once

#include "ttl/optimizer/Nodes.hpp"
#include <fmt/format.h>

namespace ttl::optimizer
{
  struct Print
  {
    void operator()(node_ptr node, fmt::memory_buffer& out) const
    {
      node.visit(*this, out);
    }

    void operator()(Binary* node, fmt::memory_buffer& out) const
    {
      fmt::format_to(out, "{}", "(");
      node->a.visit(*this, out);
      fmt::format_to(out, " {} ", node->tag);
      node->b.visit(*this, out);
      fmt::format_to(out, "{}", ")");
    }

    void operator()(Pow* node, fmt::memory_buffer& out) const
    {
      fmt::format_to(out, "{}", "(");
      node->a.visit(*this, out);
      fmt::format_to(out, "){}", node->tag);
      node->b.visit(*this, out);
    }

    void operator()(Unary* node, fmt::memory_buffer& out) const
    {
      fmt::format_to(out, "{}(", node->tag);
      node->a.visit(*this, out);
      fmt::format_to(out, "{}", ")");
    }

    void operator()(Bind* bind, fmt::memory_buffer& out) const
    {
      fmt::format_to(out, "{}(", bind->tag);
      bind->a.visit(*this, out);
      fmt::format_to(out, ",{})", bind->index);
    }

    void operator()(Partial* partial, fmt::memory_buffer& out) const
    {
      fmt::format_to(out, "{}(", partial->tag);
      partial->a.visit(*this, out);
      fmt::format_to(out, ",{})", partial->index);
    }


    void operator()(Rational* q, fmt::memory_buffer& out) const
    {
      fmt::format_to(out, "{}", q->q);
    }

    void operator()(Double* d, fmt::memory_buffer& out) const
    {
      fmt::format_to(out, "{}", d->d);
    }

    void operator()(Tensor* tensor, fmt::memory_buffer& out) const
    {
      if (tensor->index.size()) {
        fmt::format_to(out, "{}({})", *tensor->tensor, tensor->index);
      }
      else {
        fmt::format_to(out, "{}", *tensor->tensor);
      }
    }

    void operator()(Scalar* scalar, fmt::memory_buffer& out) const
    {
      if (scalar->index.size()) {
        fmt::format_to(out, "{}({})", *scalar->tensor, scalar->index);
      }
      else {
        fmt::format_to(out, "{}", *scalar->tensor);
      }
    }

    void operator()(Delta* δ, fmt::memory_buffer& out) const
    {
      fmt::format_to(out, "{}({})", δ->tag, δ->index);
    }

    void operator()(Epsilon* ε, fmt::memory_buffer& out) const
    {
      fmt::format_to(out, "{}({})", ε->tag, ε->index);
    }
  };
}
