add_library(ttl_lib5 INTERFACE)
target_include_directories(ttl_lib5 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(ttl::ttl5 ALIAS ttl_lib5)
