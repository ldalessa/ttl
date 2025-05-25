module;
#include "ttl/Equation.hpp"
#include "ttl/ExecutableSystem.hpp"
#include "ttl/Index.hpp"
#include "ttl/System.hpp"
#include "ttl/Tensor.hpp"
#include "ttl/dot.hpp"
#include "ttl/grammar.hpp"
export module ttl;

export namespace ttl
{
	using ttl::D;
	using ttl::delta;
	using ttl::dot;
	using ttl::Equation;
	using ttl::ExecutableSystem;
	using ttl::Index;
	using ttl::is_tree;
	using ttl::matrix;
	using ttl::scalar;
	using ttl::symmetrize;
	using ttl::System;
	using ttl::Tensor;
	using ttl::TensorTree;
	using ttl::vector;

	using ttl::operator+;
	using ttl::operator*;
	using ttl::operator-;
	using ttl::operator/;
}