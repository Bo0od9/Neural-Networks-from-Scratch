add_subdirectory(googletest)

add_executable(ActivationFunctionTest
        ActivationFunctionTest.cpp
)

target_include_directories(ActivationFunctionTest PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/eigen
        ${CMAKE_SOURCE_DIR}/EigenRand
        ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include)

target_link_libraries(ActivationFunctionTest
        NeuralNetworkLibrary
        gtest
        gtest_main
        pthread)

add_test(NAME ActivationFunctionTest COMMAND ActivationFunctionTest)

add_executable(LossFunctionTest
        LossFunctionTest.cpp)

target_include_directories(LossFunctionTest PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/eigen
        ${CMAKE_SOURCE_DIR}/EigenRand
        ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include)

target_link_libraries(LossFunctionTest
        NeuralNetworkLibrary
        gtest
        gtest_main
        pthread)

add_test(NAME LossFunctionTest COMMAND LossFunctionTest)

add_executable(LayerTest
        LayerTest.cpp)

target_include_directories(LayerTest PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/eigen
        ${CMAKE_SOURCE_DIR}/EigenRand
        ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include)

target_link_libraries(LayerTest
        NeuralNetworkLibrary
        gtest
        gtest_main
        pthread)

add_test(NAME LayerTest COMMAND LayerTest)

add_executable(NetworkTest
        NetworkTest.cpp)

target_include_directories(NetworkTest PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/eigen
        ${CMAKE_SOURCE_DIR}/EigenRand
        ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include)

target_link_libraries(NetworkTest
        NeuralNetworkLibrary
        gtest
        gtest_main
        pthread)

add_test(NAME NetworkTest COMMAND NetworkTest)
