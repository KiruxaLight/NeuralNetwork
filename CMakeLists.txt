cmake_minimum_required(VERSION 3.26)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

# Это для основного приложения
add_executable(NeuralNetwork main.cpp Model.cpp LossFunction.cpp DataLoader.cpp Architecture.cpp
    ActivationFunction.cpp LinearLayer.cpp
        Metrics.h
        Metrics.cpp)
include_directories(${CMAKE_SOURCE_DIR}/libs/eigen-3.4.0)

# Это для отдельного исполняемого файла теста
add_executable(test_dataloader tests/test_dataloader.cpp DataLoader.cpp)
include_directories(${CMAKE_SOURCE_DIR}/libs/eigen-3.4.0)
