cmake_minimum_required(VERSION 3.5)
project(Majsan3k)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(Majsan3k ${SOURCE_FILES})
target_link_libraries(Majsan3k ${ARMADILLO_LIBRARIES})
