cmake_minimum_required(VERSION 2.8)
project (tkDNN)

set(BUILD_DEPS true CACHE BOOL "If true download deps")

if( ${BUILD_DEPS} )
	message("Launching pre-build dependency installer script...")

    execute_process (COMMAND bash -c "bash build_models.sh download"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

	set(BUILD_DEPS false CACHE BOOL "If true download deps" FORCE)
	message("Finished dowloading test weights")
endif()

if(DEBUG)
    add_definitions(-DDEBUG)
endif()

find_package(CUDA QUIET REQUIRED)
find_package(OpenCV QUIET)
if(${OpenCV_FOUND})
    message("Compiling with openCV support")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")
else()
    message(WARNING "OpenCV not found, compiling without it")
endif()

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
cuda_add_library(kernels SHARED src/kernels/activation_elu.cu 
                                src/kernels/activation_leaky.cu
                                src/kernels/activation_logistic.cu
                                src/kernels/reorg.cu
                                src/kernels/softmax.cu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS})
add_library(tkDNN SHARED src/Layer.cpp src/LayerWgs.cpp
    src/Dense.cpp src/Activation.cpp src/Conv2d.cpp src/Flatten.cpp src/MulAdd.cpp src/Pooling.cpp src/Softmax.cpp
    src/Route.cpp src/Reorg.cpp src/Region.cpp src/Network.cpp src/utils.cpp src/NetworkRT.cpp)
target_link_libraries(tkDNN kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} -lcudnn -lnvinfer ${OpenCV_LIBS})

add_executable(test_simple tests/simple/test_simple.cpp)
target_link_libraries(test_simple tkDNN)

add_executable(test_mnist tests/mnist/test_mnist.cpp)
target_link_libraries(test_mnist tkDNN)

add_executable(test_mnistRT tests/mnist/test_mnistRT.cpp)
target_link_libraries(test_mnistRT tkDNN)

add_executable(test_yolo tests/yolo/yolo.cpp)
target_link_libraries(test_yolo tkDNN)

add_executable(test_yolo_tiny tests/yolo-tiny/yolo-tiny.cpp)
target_link_libraries(test_yolo_tiny tkDNN)

add_executable(test_rtinference tests/test_rtinference/rtinference.cpp)
target_link_libraries(test_rtinference tkDNN)