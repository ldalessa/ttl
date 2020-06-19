# add_executable(constexpr_tests constexpr_tests.cpp)
# target_link_libraries(constexpr_tests PRIVATE ttl::ttl)

add_executable(burgers burgers.cpp)
target_link_libraries(burgers PRIVATE ttl::ttl)

add_executable(sedov sedov.cpp)
target_link_libraries(sedov PRIVATE ttl::ttl)

add_executable(burgers2 burgers2.cpp)
target_link_libraries(burgers2 PRIVATE ttl::ttl2)

add_executable(sedov2 sedov2.cpp)
target_link_libraries(sedov2 PRIVATE ttl::ttl2)
