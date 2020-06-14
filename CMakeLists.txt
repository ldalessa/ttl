cmake_minimum_required(VERSION 3.18)
project(ttl CXX)

# -----------------------------------------------------------------------------
# Set our basic C++ standards
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQURED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

add_subdirectory(include)
add_subdirectory(test)
