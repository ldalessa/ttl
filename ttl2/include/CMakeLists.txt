add_library(ttl_lib2 INTERFACE)
target_include_directories(ttl_lib2 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(ttl::ttl2 ALIAS ttl_lib2)
