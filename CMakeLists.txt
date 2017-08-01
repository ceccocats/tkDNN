cmake_minimum_required(VERSION 2.8)

project (tkDNN)

find_package(CUDA QUIET REQUIRED)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
cuda_add_library(kernels SHARED src/kernels/activation_elu.cu 
                                src/kernels/activation_leaky.cu
                                src/kernels/activation_logistic.cu
                                src/kernels/reorg.cu
                                src/kernels/softmax.cu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
add_library(tkDNN SHARED src/Layer.cpp src/LayerWgs.cpp
    src/Dense.cpp src/Activation.cpp src/Conv2d.cpp src/Flatten.cpp src/MulAdd.cpp src/Pooling.cpp src/Softmax.cpp
    src/Route.cpp src/Reorg.cpp src/Region.cpp src/Network.cpp src/utils.cpp)
target_link_libraries(tkDNN kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} -lcudnn -lnvinfer)

add_executable(test_simple tests/test/test.cpp)
target_link_libraries(test_simple tkDNN)

add_executable(test_mnist tests/mnist/test.cpp)
target_link_libraries(test_mnist tkDNN)

add_executable(test_yolo tests/yolo/yolo.cpp)
target_link_libraries(test_yolo tkDNN)