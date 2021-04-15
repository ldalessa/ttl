cmake_minimum_required(VERSION 3.18)
project(ttl CXX)

set(TTL_SANITIZE "" CACHE STRING "-fsanitize compiler options")
if (TTL_SANITIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${TTL_SANITIZE}")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=${TTL_SANITIZE}")
endif (TTL_SANITIZE)

# -----------------------------------------------------------------------------
# External project dependencies
# -----------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(fmt
  GIT_REPOSITORY     https://github.com/fmtlib/fmt.git
  GIT_TAG            master)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(ce
  GIT_REPOSITORY     git@github.com:ldalessa/ce.git
  GIT_TAG            main)
FetchContent_MakeAvailable(ce)

FetchContent_Declare(ofw
  GIT_REPOSITORY     https://github.com/jfalcou/ofw.git
  GIT_TAG            main)
set(OFW_BUILD_TEST OFF CACHE INTERNAL "OFF")
FetchContent_MakeAvailable(ofw)

FetchContent_Declare(eve
  GIT_REPOSITORY     https://github.com/jfalcou/eve.git
  GIT_TAG            develop)
FetchContent_MakeAvailable(eve)

add_library(ttl_lib INTERFACE)
target_include_directories(ttl_lib INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(ttl_lib INTERFACE ce::ce eve::eve fmt::fmt ofw::ofw)
target_compile_features(ttl_lib INTERFACE cxx_std_20)
add_library(ttl::ttl ALIAS ttl_lib)

add_subdirectory(examples)
add_subdirectory(tests)
