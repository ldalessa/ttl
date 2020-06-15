# add_executable(constexpr_tests constexpr_tests.cpp)
# target_link_libraries(constexpr_tests PRIVATE ttl::ttl)

add_executable(sedov sedov.cpp)
target_link_libraries(sedov PRIVATE ttl::ttl)
