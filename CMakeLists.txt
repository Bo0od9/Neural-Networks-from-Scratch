cmake_minimum_required(VERSION 3.14)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

add_compile_options(-Wall)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_MAIN_APP "Build main application" ON)

add_subdirectory(src)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
