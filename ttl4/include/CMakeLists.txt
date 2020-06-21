add_library(ttl_lib4 INTERFACE)
target_include_directories(ttl_lib4 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(ttl::ttl4 ALIAS ttl_lib4)
