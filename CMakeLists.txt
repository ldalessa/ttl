cmake_minimum_required(VERSION 3.18)
project(ttl CXX)

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

FetchContent_Declare(raberu
  GIT_REPOSITORY     https://github.com/jfalcou/raberu.git
  GIT_TAG            main)
set(RABERU_BUILD_TEST OFF CACHE INTERNAL "OFF")
FetchContent_MakeAvailable(raberu)

add_library(ttl_lib INTERFACE)
target_include_directories(ttl_lib INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_definitions(ttl_lib INTERFACE TTL_MAX_PARSE_INDEX=${TTL_MAX_PARSE_INDEX})
target_link_libraries(ttl_lib INTERFACE kumi::kumi raberu::raberu)
target_compile_features(ttl_lib INTERFACE cxx_std_20)
add_library(ttl::ttl ALIAS ttl_lib)

add_subdirectory(examples)
add_subdirectory(tests)
