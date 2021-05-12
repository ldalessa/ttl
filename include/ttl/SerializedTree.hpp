#pragma once

#include "ttl/Nodes.hpp"
#include "ttl/ParseTree.hpp"

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
          c.a = nodes + a;
          c.b = nodes + b;
          return ++i;
        },
        [&]<parse::unary_node_t Unary>(Unary&& n) -> int {
          int a = i = serialize(*n.a, i);
          auto& b = nodes[i] = n;
          b.a = nodes + a;
          return ++i;
        },
        [&]<parse::leaf_node_t Leaf>(Leaf&&) -> int {
          nodes[i] = n;
          return ++i;
        }
      };

      return visit(n, op);
    }
  };
}
