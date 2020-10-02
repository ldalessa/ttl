// from https://en.cppreference.com/w/cpp/utility/variant/visit
#pragma once

namespace ttl::mp {
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
}
