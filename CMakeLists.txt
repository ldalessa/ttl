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

add_library(ttl_lib INTERFACE)
target_include_directories(ttl_lib INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_definitions(ttl_lib INTERFACE TTL_MAX_PARSE_INDEX=${TTL_MAX_PARSE_INDEX})
target_link_libraries(ttl_lib INTERFACE kumi::kumi)
target_compile_features(ttl_lib INTERFACE cxx_std_26)
add_library(ttl::ttl ALIAS ttl_lib)

add_subdirectory(examples)
add_subdirectory(tests)
