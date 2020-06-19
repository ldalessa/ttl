cmake_minimum_required(VERSION 3.18)
project(ttl CXX)

# -----------------------------------------------------------------------------
# Set our basic C++ standards
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQURED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TTL_SANITIZE "" CACHE STRING "-fsanitize compiler options")
if (TTL_SANITIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${TTL_SANITIZE}")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=${TTL_SANITIZE}")
endif (TTL_SANITIZE)

include(FetchContent)

add_subdirectory(include)
add_subdirectory(test)
