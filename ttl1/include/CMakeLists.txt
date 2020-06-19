add_library(ttl_lib1 INTERFACE)
target_include_directories(ttl_lib1 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(ttl::ttl1 ALIAS ttl_lib1)
