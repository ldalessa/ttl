add_library(ttl_lib INTERFACE)
target_include_directories(ttl_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(ttl::ttl ALIAS ttl_lib)
