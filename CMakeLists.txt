cmake_minimum_required(VERSION 4.0.2)

set(CMAKE_CXX_FLAGS "-stdlib=libc++")
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "a9e1cf81-9932-4810-974b-6eccaf14e457")
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_MODULE_STD 1)
project(ttl CXX)
unset(CMAKE_CXX_FLAGS)

set(TTL_MAX_PARSE_INDEX "16" CACHE STRING "The largest tensor index supported in the expression parser")

# -----------------------------------------------------------------------------
# External project dependencies
# -----------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(kumi
  GIT_REPOSITORY     https://github.com/jfalcou/kumi.git
  GIT_TAG            main)
set(KUMI_BUILD_TEST OFF CACHE INTERNAL "OFF")
FetchContent_MakeAvailable(kumi)

add_library(ttl_impl INTERFACE)
target_include_directories(ttl_impl INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_definitions(ttl_impl INTERFACE TTL_MAX_PARSE_INDEX=${TTL_MAX_PARSE_INDEX})
target_link_libraries(ttl_impl INTERFACE kumi::kumi)
target_compile_features(ttl_impl INTERFACE cxx_std_26)
target_compile_options(ttl_impl INTERFACE
  --include=ttl/FWD.hpp
  --include=ttl/ARROW.hpp)

add_library(ttl_mod)
target_sources(ttl_mod PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES include/ttl/ttl.cppm)
target_compile_features(ttl_mod PUBLIC cxx_std_26)
target_link_libraries(ttl_mod PUBLIC ttl_impl)

# add_subdirectory(examples)
add_subdirectory(tests)
