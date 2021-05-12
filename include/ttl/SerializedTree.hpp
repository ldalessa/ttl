#pragma once

#include "ttl/Nodes.hpp"
#include "ttl/ParseTree.hpp"
#include "ttl/cpos.hpp"
#include <fmt/format.h>

namespace ttl
{
  template <int M>
  struct SerializedTree
  {
    parse::AnyNode nodes[M]{};

    constexpr SerializedTree(parse::Tree<M>&& tree)
    {
      serialize(*tree.root, 0);
    }

    constexpr auto serialize(parse::Node const& n, int i) -> int
    {
      parse::overloaded op = {
        [&]<parse::binary_node_t Binary>(Binary&& n) -> int {
          int a = i = serialize(*n.a, i);
          int b = i = serialize(*n.b, i);
          auto& c = nodes[i] = n;
          c.a = nodes + a - 1;
          c.b = nodes + b - 1;
          return ++i;
        },
        [&]<parse::unary_node_t Unary>(Unary&& n) -> int {
          int a = i = serialize(*n.a, i);
          auto& b = nodes[i] = n;
          b.a = nodes + a - 1;
          return ++i;
        },
        [&]<parse::leaf_node_t Leaf>(Leaf&&) -> int {
          nodes[i] = n;
          return ++i;
        }
      };

      return visit(n, op);
    }

    friend auto tag_invoke(dot_tag_, SerializedTree const& tree, fmt::memory_buffer &out)
      -> fmt::memory_buffer&
    {
      return out;
    }

    void print(fmt::memory_buffer &out) const
    {
      return nodes[M-1].print(out);
    }
  };
}
