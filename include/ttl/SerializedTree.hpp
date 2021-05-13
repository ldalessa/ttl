#pragma once

#include "ttl/Nodes.hpp"
#include "ttl/ParseTree.hpp"
#include "ttl/cpos.hpp"
#include <ce/dvector.hpp>
#include <fmt/format.h>

namespace ttl
{
  template <int M>
  struct SerializedTree
  {
    nodes::AnyNode nodes[M]{};

    constexpr SerializedTree(ParseTree<M>&& tree)
    {
      serialize(*tree.root, 0);
    }

    constexpr auto id() const -> std::string_view
    {
      return nodes[M-1].id();
    }

    constexpr auto tag() const -> TreeTag
    {
      return nodes[M-1].tag();
    }

    constexpr auto outer_index() const -> TensorIndex
    {
      return nodes[M-1].outer_index();
    }

    constexpr auto serialize(nodes::Node const& n, int i) -> int
    {
      nodes::overloaded op = {
        [&]<nodes::binary_node_t Binary>(Binary&& n) -> int {
          int a = i = serialize(*n.a, i);
          int b = i = serialize(*n.b, i);
          auto& c = nodes[i] = n;
          c.a = nodes + a - 1;
          c.b = nodes + b - 1;
          return ++i;
        },
        [&]<nodes::unary_node_t Unary>(Unary&& n) -> int {
          int a = i = serialize(*n.a, i);
          auto& b = nodes[i] = n;
          b.a = nodes + a - 1;
          return ++i;
        },
        [&]<nodes::leaf_node_t Leaf>(Leaf&&) -> int {
          nodes[i] = n;
          return ++i;
        }
      };

      return visit(n, op);
    }

    void dot(fmt::memory_buffer &out) const
    {
      ce::dvector<int> stack;
      for (int i = 0; i < M; ++i) {
        visit(nodes[i], nodes::overloaded {
            [&]<nodes::binary_node_t Binary>(Binary&& node) {
              node.dot(out, i, stack.pop_back(), stack.pop_back());
            },
            [&]<nodes::unary_node_t Unary>(Unary&& node) {
              node.dot(out, i, stack.pop_back());
            },
            [&]<nodes::leaf_node_t Leaf>(Leaf&& node) {
              node.dot(out, i);
            }
          });
        stack.push_back(i);
      }
    }

    void print(fmt::memory_buffer &out) const
    {
      return nodes[M-1].print(out);
    }
  };
}
