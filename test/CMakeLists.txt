# add_executable(constexpr_tests constexpr_tests.cpp)
# target_link_libraries(constexpr_tests PRIVATE ttl::ttl)

add_executable(burgers1 burgers.cpp)
target_link_libraries(burgers1 PRIVATE ttl::ttl1)

add_executable(sedov1 sedov.cpp)
target_link_libraries(sedov1 PRIVATE ttl::ttl1)

add_executable(burgers2 burgers.cpp)
target_link_libraries(burgers2 PRIVATE ttl::ttl2)

add_executable(sedov2 sedov.cpp)
target_link_libraries(sedov2 PRIVATE ttl::ttl2)

add_executable(burgers3 burgers.cpp)
target_link_libraries(burgers3 PRIVATE ttl::ttl3)

add_executable(sedov3 sedov.cpp)
target_link_libraries(sedov3 PRIVATE ttl::ttl3)

add_executable(burgers4 burgers.cpp)
target_link_libraries(burgers4 PRIVATE ttl::ttl4)

add_executable(sedov4 sedov.cpp)
target_link_libraries(sedov4 PRIVATE ttl::ttl4)
