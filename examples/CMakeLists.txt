include(FetchContent)
FetchContent_Declare(CLI11
  GIT_REPOSITORY     https://github.com/CLIUtils/CLI11.git
  GIT_TAG            main)
FetchContent_MakeAvailable(CLI11)

add_executable(burgers burgers.cpp)
target_link_libraries(burgers PRIVATE ttl::ttl)

add_executable(ns ns.cpp)
target_link_libraries(ns PRIVATE ttl::ttl CLI11::CLI11)