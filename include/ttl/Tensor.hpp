#pragma once

#include "ttl/Nodes.hpp"
#include "ttl/ParseTree.hpp"
#include "ttl/TensorIndex.hpp"
#include "ttl/TensorRef.hpp"
#include <kumi.hpp>

namespace ttl
{
  template <class... Trees>
  struct Tensor : TensorRef
  {
    std::string_view id_ = {};
    int rank_ = 0;
    kumi::tuple<Trees...> trees_;

    constexpr Tensor(std::string_view id, int rank)
        : id_(id)
        , rank_(rank)
    {
    }

    constexpr Tensor(parse_tree auto&& tree)
        : id_    { tree.root->id() }
        , rank_  { tree.root->rank() }
        , trees_ { std::move(tree) }
    {
      trees_([](auto const& tree) {
        assert(tree.tag() != SCALAR);
      });
    }

    constexpr auto id() const -> std::string_view override
    {
      return id_;
    }

    constexpr auto n_trees() const -> int override
    {
      return sizeof...(Trees);
    }

    constexpr auto rank() const -> int override
    {
      return rank_;
    }

    constexpr auto outer_index() const -> TensorIndex override
    {
      return trees_([](auto const&... trees) {
        return (trees.outer_index() + ... + TensorIndex{});
      });
    }

    constexpr auto promote_tensor(TensorIndex index) const -> ParseTree<2>
    {
      return ParseTree<2>
      {
        new nodes::Bind {
          new nodes::Tensor { this },
          index
        }
      };
    }

    constexpr auto operator()(std::same_as<TensorIndex> auto... is) const
    {
      return promote_tensor({is...});
    }

    void print(fmt::memory_buffer &out, bool follow_links) const override
    {
      if constexpr (sizeof...(Trees) == 0) {
        fmt::format_to(out, "{}", id());
      }
      else {
        trees_([&](auto const&... trees) {
          (trees.print(out, follow_links), ...);
        });
      }
    }

    auto dot(fmt::memory_buffer &out, int i) const -> int override
    {
      if constexpr (sizeof...(Trees) == 0) {
        fmt::format_to(out, "\tnode{}[label=\"{}\"]\n", i, id());
      }
      else {
        trees_([&](auto const&... trees) {
          (((i = trees.dot(out, i)), ++i), ...);
        });
      }
      return i - 1;
    }
  };

  Tensor(parse_tree auto&& tree) -> Tensor<decltype(SerializedTree(std::move(tree)))>;

  constexpr auto scalar(std::string_view id) -> ttl::Tensor<>
  {
    return { id, 0 };
  }

  constexpr auto vector(std::string_view id) -> ttl::Tensor<>
  {
    return { id, 1 };
  }

  constexpr auto matrix(std::string_view id) -> ttl::Tensor<>
  {
    return { id, 1 };
  }
}
