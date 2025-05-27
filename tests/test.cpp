#include <print>
import ttl;

namespace
{
	constexpr ttl::tensor A = ttl::matrix("A");
	constexpr ttl::tensor B = ttl::matrix("B");
	// constexpr ttl::tensor C = ttl::matrix("C");

	constexpr ttl::index i = "i";
	constexpr ttl::index p = 'p';

	constexpr auto a = A(i, p);
	constexpr auto b = B(p, i);
	constexpr auto c = a + b;
	constexpr auto d = c(i, p);
	constexpr auto e = D(d, i, p);

	// constexpr ttl::System test = {
	// 	C <<= A(i, j) + B(i, j)
	// };

	// [[maybe_unused]] constexpr ttl::ExecutableSystem<double, 3, test> test3d;
}

int main()
{
	std::println("{}", a);
	std::println("{}", b);
	std::println("{}", c);
	return 0;
}
