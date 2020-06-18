cmake_minimum_required(VERSION 3.5)

project (tkDNN)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wno-deprecated-declarations -Wno-unused-variable")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tkDNN)

# project specific flags
if(DEBUG)
    add_definitions(-DDEBUG)
endif()

add_definitions(-DTKDNN_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDA 9.0 REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 --compiler-options '-fPIC'")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)

find_package(CUDNN REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})


# compile
file(GLOB tkdnn_CUSRC "src/kernels/*.cu" "src/sorting.cu")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})
cuda_add_library(kernels SHARED ${tkdnn_CUSRC})


#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

# gives problems in cross-compiling, probably malformed cmake config
#find_package(yaml-cpp REQUIRED)

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES})
add_library(tkDNN SHARED ${tkdnn_SRC})
target_link_libraries(tkDNN ${tkdnn_LIBS})

#static 
#add_library(tkDNN_static STATIC ${tkdnn_SRC})
#target_link_libraries(tkDNN_static ${tkdnn_LIBS})

# SMALL NETS
add_executable(test_simple tests/simple/test_simple.cpp)
target_link_libraries(test_simple tkDNN)

add_executable(test_mnist tests/mnist/test_mnist.cpp)
target_link_libraries(test_mnist tkDNN)

add_executable(test_mnistRT tests/mnist/test_mnistRT.cpp)
target_link_libraries(test_mnistRT tkDNN)

add_executable(test_imuodom tests/imuodom/imuodom.cpp)
target_link_libraries(test_imuodom tkDNN)

# DARKNET
file(GLOB darknet_SRC "tests/darknet/*.cpp")
foreach(test_SRC ${darknet_SRC})
    get_filename_component(test_NAME "${test_SRC}" NAME_WE)
    set(test_NAME test_${test_NAME})
    add_executable(${test_NAME} ${test_SRC})
    target_link_libraries(${test_NAME} tkDNN)
endforeach()

# MOBILENET
add_executable(test_mobilenetv2ssd tests/mobilenet/mobilenetv2ssd/mobilenetv2ssd.cpp)
target_link_libraries(test_mobilenetv2ssd tkDNN)

add_executable(test_bdd-mobilenetv2ssd tests/mobilenet/bdd-mobilenetv2ssd/bdd-mobilenetv2ssd.cpp)
target_link_libraries(test_bdd-mobilenetv2ssd tkDNN)

add_executable(test_mobilenetv2ssd512 tests/mobilenet/mobilenetv2ssd512/mobilenetv2ssd512.cpp)
target_link_libraries(test_mobilenetv2ssd512 tkDNN)

# BACKBONES
add_executable(test_resnet101 tests/backbones/resnet101/resnet101.cpp)
target_link_libraries(test_resnet101 tkDNN)

add_executable(test_dla34 tests/backbones/dla34/dla34.cpp)
target_link_libraries(test_dla34 tkDNN)

# CENTERNET
add_executable(test_resnet101_cnet tests/centernet/resnet101_cnet/resnet101_cnet.cpp)
target_link_libraries(test_resnet101_cnet tkDNN)

add_executable(test_dla34_cnet tests/centernet/dla34_cnet/dla34_cnet.cpp)
target_link_libraries(test_dla34_cnet tkDNN)

# DEMOS
add_executable(test_rtinference tests/test_rtinference/rtinference.cpp)
target_link_libraries(test_rtinference tkDNN)

add_executable(map_demo demo/demo/map.cpp)
target_link_libraries(map_demo tkDNN)

add_executable(demo demo/demo/demo.cpp)
target_link_libraries(demo tkDNN)

#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------
#if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" 
#         CACHE PATH "default install path" FORCE)
#endif()
message("install dir:" ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY include/ DESTINATION include/)
install(TARGETS tkDNN kernels DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" # source directory
        DESTINATION "share/tkDNN/cmake/" # target directory
)

