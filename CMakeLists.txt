cmake_minimum_required(VERSION 2.8)

project (tkDNN)

find_package(CUDA QUIET REQUIRED)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
add_library(tkDNN SHARED src/Layer.cpp src/LayerWgs.cpp src/Dense.cpp src/Activation.cpp src/Network.cpp src/utils.cpp)

add_executable(tkDNNtest tests/test.cpp)
message(${CUDA_LIBRARIES})
target_link_libraries(tkDNNtest tkDNN ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} /usr/local/cuda-8.0/cudnn/libcudnn.so)