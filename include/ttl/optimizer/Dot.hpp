#pragma once

#include "ttl/Tag.hpp"
#include "ttl/optimizer/Nodes.hpp"
#include <fmt/format.h>

namespace ttl::optimizer
{
  struct Dot
  {
    int i = 0;
    fmt::memory_buffer out;

    void format(auto const&... args) {
      fmt::format_to(out, args...);
    }

    void write(FILE* file) const
    {
      std::fwrite(out.data(), out.size(), 1, file);
    }

    void operator()(node_ptr const& node)
    {
      visit(node, *this);
    }

    auto operator()(tags::binary, node_ptr const& node) -> int
    {
      int a = visit(node->a(), *this);
      int b = visit(node->b(), *this);
      Index outer = node->outer();
      if (outer.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{} ↑{}\"]\n", i, node->tag, outer);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, node->tag, outer);
      }
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
      fmt::format_to(out, "\tnode{} -- node{}\n", i, b);
      return i++;
    }

    auto operator()(tags::unary, node_ptr const& node) -> int
    {
      int a = visit(node->a(), *this);
      fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, node->tag);
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
      return i++;
    }

    auto operator()(tags::binder, node_ptr const& bind) -> int
    {
      int a = visit(bind->a(), *this);

      Index outer = bind->outer();
      Index child = bind->a()->outer();
      if (outer.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{}({},{}) ↑{}\"]\n", i, bind->tag, child, bind->tensor_index, outer);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}({},{})\"]\n", i, bind->tag, child, bind->tensor_index);
      }
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
      return i++;
    }

    auto operator()(tags::leaf, node_ptr const& leaf) -> int
    {
      fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, leaf->tag, leaf->tensor_index);
      return i++;
    }

    auto operator()(tags::immediate, node_ptr const& lit) -> int
    {
      if (lit->tag == DOUBLE) {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, lit->d * as<double>(lit->q));
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, lit->q);
      }
      return i++;
    }

    auto operator()(tags::tensor, node_ptr const& tensor) -> int
    {
      if (tensor->tensor_index.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, *tensor->tensor, tensor->tensor_index);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, *tensor->tensor);
      }
      return i++;
    }

    auto operator()(tags::scalar, node_ptr const& scalar) -> int
    {
      if (scalar->scalar_index.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, *scalar->tensor, scalar->scalar_index);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, *scalar->tensor);
      }
      return i++;
    }
  };
}
