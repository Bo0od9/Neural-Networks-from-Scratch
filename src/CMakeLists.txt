add_library(NeuralNetworkLibrary STATIC
        declarations.h
        Layer.h
        Layer.cpp
        Network.h
        Network.cpp
        ActivationFunction.h
        LossFunction.h
        SigmoidActivation.h
        TanhActivation.h
        ReluActivation.h
        SoftmaxActivation.h
        SoftmaxActivation.cpp
        MeanSquaredError.h
        MeanSquaredError.cpp
        CrossEntropyLoss.h
        CrossEntropyLoss.cpp
        MNISTLoader.h
        MNISTLoader.cpp)

target_include_directories(NeuralNetworkLibrary PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/eigen
        ${CMAKE_SOURCE_DIR}/EigenRand)

if (BUILD_MAIN_APP)
    add_executable(NeuralNetwork main.cpp)

    target_include_directories(NeuralNetwork PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/eigen
            ${CMAKE_SOURCE_DIR}/EigenRand)

    target_link_libraries(NeuralNetwork NeuralNetworkLibrary)
endif ()