cmake_minimum_required(VERSION 2.8)

project (tkDNN)

find_package(CUDA QUIET REQUIRED)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
cuda_add_library(kernels SHARED src/kernels/activation_elu.cu)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
add_library(tkDNN SHARED src/Layer.cpp src/LayerWgs.cpp
    src/Dense.cpp src/Activation.cpp src/Conv2d.cpp src/Flatten.cpp src/MulAdd.cpp src/Pooling.cpp src/Softmax.cpp
    src/Network.cpp src/utils.cpp)
target_link_libraries(tkDNN kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} -lcudnn)

add_executable(tkDNNtest tests/test.cpp)
target_link_libraries(tkDNNtest tkDNN)

add_executable(mnist tests/mnist/test.cpp)
target_link_libraries(mnist tkDNN)
