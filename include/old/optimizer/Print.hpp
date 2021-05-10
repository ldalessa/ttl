#pragma once

#include "ttl/Tag.hpp"
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

    auto operator()(node_ptr const& node) -> auto&
    {
      visit(node, *this);
      return out;
    }

    void operator()(tags::binary, node_ptr const& node)
    {
      out.append("("sv);
      visit(node->a, *this);
      fmt::format_to(out, " {} ", node->tag);
      visit(node->b, *this);
      out.append(")"sv);
    }

    void operator()(tags::pow, node_ptr const& node)
    {
      out.append("("sv);
      visit(node->a, *this);
      fmt::format_to(out, "){}", node->tag);
      visit(node->b, *this);
    }

    void operator()(tags::unary, node_ptr const& node)
    {
      fmt::format_to(out, "{}(", node->tag);
      visit(node->a, *this);
      out.append(")"sv);
    }

    void operator()(tags::binder, node_ptr const& bind)
    {
      fmt::format_to(out, "{}(", bind->tag);
      visit(bind->a, *this);
      fmt::format_to(out, ",{})", bind->tensor_index);
    }

    void operator()(tags::leaf, node_ptr const& i)
    {
      fmt::format_to(out, "{}({})", i->tag, i->tensor_index);
    }

    void operator()(tags::immediate, node_ptr const& i)
    {
      if (i->tag == DOUBLE) {
        fmt::format_to(out, "{}", i->d * as<double>(i->q));
      }
      else {
        fmt::format_to(out, "{}", i->q);
      }
    }

    void operator()(tags::tensor, node_ptr const& tensor)
    {
      if (tensor->tensor_index.size()) {
        fmt::format_to(out, "{}({})", *tensor->tensor, tensor->tensor_index);
      }
      else {
        fmt::format_to(out, "{}", *tensor->tensor);
      }
    }

    void operator()(tags::scalar, node_ptr const& scalar)
    {
      if (scalar->scalar_index.size()) {
        fmt::format_to(out, "{}({})", *scalar->tensor, scalar->scalar_index);
      }
      else {
        fmt::format_to(out, "{}", *scalar->tensor);
      }
    }
  };
}
