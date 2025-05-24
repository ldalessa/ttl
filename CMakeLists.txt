cmake_minimum_required(VERSION 3.18)
project(ttl CXX)

set(TTL_SANITIZE "" CACHE STRING "-fsanitize compiler options")
if (TTL_SANITIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${TTL_SANITIZE}")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=${TTL_SANITIZE}")
endif (TTL_SANITIZE)

set(TTL_MAX_PARSE_INDEX "16" CACHE STRING "The largest tensor index supported in the expression parser")

# -----------------------------------------------------------------------------
# External project dependencies
# -----------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(fmt
  GIT_REPOSITORY     https://github.com/fmtlib/fmt.git
  GIT_TAG            master)
FetchContent_MakeAvailable(fmt)

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

FetchContent_Declare(eve
  GIT_REPOSITORY     https://github.com/jfalcou/eve.git
  GIT_TAG            main)
FetchContent_MakeAvailable(eve)

add_library(ttl_lib INTERFACE)
target_include_directories(ttl_lib INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_definitions(ttl_lib INTERFACE TTL_MAX_PARSE_INDEX=${TTL_MAX_PARSE_INDEX})
target_link_libraries(ttl_lib INTERFACE eve::eve fmt::fmt kumi::kumi raberu::raberu)
target_compile_features(ttl_lib INTERFACE cxx_std_20)
add_library(ttl::ttl ALIAS ttl_lib)

add_subdirectory(examples)
add_subdirectory(tests)
