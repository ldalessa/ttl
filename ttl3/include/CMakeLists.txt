add_library(ttl_lib3 INTERFACE)
target_include_directories(ttl_lib3 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(ttl::ttl3 ALIAS ttl_lib3)
