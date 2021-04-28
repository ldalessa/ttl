#pragma once

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

    void operator()(node_ptr node)
    {
      node.visit(*this);
    }

    auto operator()(Binary* node) -> int
    {
      int a = node->a.visit(*this);
      int b = node->b.visit(*this);
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

    auto operator()(Unary* node) -> int
    {
      int a = node->a.visit(*this);
      fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, node->tag);
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
      return i++;
    }

    auto operator()(Bind* bind) -> int
    {
      int a = bind->a.visit(*this);

      Index outer = bind->outer();
      Index child = bind->a->outer();
      if (outer.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{}({},{}) ↑{}\"]\n", i, bind->tag, child, bind->index, outer);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}({},{})\"]\n", i, bind->tag, child, bind->index);
      }
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
      return i++;
    }

    auto operator()(Partial* partial) -> int
    {
      int a = partial->a.visit(*this);

      Index outer = partial->outer();
      Index child = partial->a->outer();
      if (outer.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{}({},{}) ↑{}\"]\n", i, partial->tag, child, partial->index, outer);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}({},{})\"]\n", i, partial->tag, child, partial->index);
      }
      fmt::format_to(out, "\tnode{} -- node{}\n", i, a);
      return i++;
    }

    auto operator()(Rational* q) -> int
    {
      fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, q->q);
      return i++;
    }

    auto operator()(Double* d) -> int
    {
      fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, d->d);
      return i++;
    }

    auto operator()(Tensor* tensor) -> int
    {
      if (tensor->index.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, *tensor->tensor, tensor->index);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, *tensor->tensor);
      }
      return i++;
    }

    auto operator()(Scalar* scalar) -> int
    {
      if (scalar->index.size()) {
        fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, *scalar->tensor, scalar->index);
      }
      else {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, *scalar->tensor);
      }
      return i++;
    }

    auto operator()(Delta* δ) -> int
    {
      fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, δ->tag, δ->index);
      return i++;
    }

    auto operator()(Epsilon* ε) -> int
    {
      fmt::format_to(out, "\tnode{}[label=\"{}({})\"]\n", i, ε->tag, ε->index);
      return i++;
    }
  };
}
