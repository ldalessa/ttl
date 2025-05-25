import ttl;

namespace
{
	constexpr ttl::Tensor A = ttl::matrix("A");
	constexpr ttl::Tensor B = ttl::matrix("B");
	constexpr ttl::Tensor C = ttl::matrix("C");

	constexpr ttl::Index i = 'i';
	constexpr ttl::Index j = 'j';

	constexpr ttl::System test = {
		C <<= A(i, j) + B(i, j)
	};

	[[maybe_unused]] constexpr ttl::ExecutableSystem<double, 3, test> test3d;
}

int main()
{
	return 0;
}
