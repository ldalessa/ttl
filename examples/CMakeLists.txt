FetchContent_Declare(docopt
  GIT_REPOSITORY     https://github.com/docopt/docopt.cpp.git
  GIT_TAG            master)
FetchContent_MakeAvailable(docopt)

add_executable(burgers burgers.cpp)
target_link_libraries(burgers PRIVATE ttl::ttl)

add_executable(ns ns.cpp)
target_link_libraries(ns PRIVATE ttl::ttl docopt)
