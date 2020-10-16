add_executable(burgers burgers.cpp)
target_link_libraries(burgers PRIVATE ttl::ttl)

add_executable(sedov sedov.cpp)
target_link_libraries(sedov PRIVATE ttl::ttl)
