cmake_minimum_required(VERSION 2.8)

project (tkDNN)

find_package(CUDA QUIET REQUIRED)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
cuda_add_library(kernels SHARED src/kernels/activation_elu.cu)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
add_library(tkDNN SHARED src/Layer.cpp src/LayerWgs.cpp 
    src/Dense.cpp src/Activation.cpp src/Conv2d.cpp src/Conv3d.cpp src/Flatten.cpp
    src/Network.cpp src/utils.cpp)
target_link_libraries(tkDNN kernels)

add_executable(tkDNNtest tests/test.cpp)
message(${CUDA_LIBRARIES})
target_link_libraries(tkDNNtest tkDNN 
    ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib/libcudnn.so)