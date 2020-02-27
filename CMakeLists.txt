cmake_minimum_required(VERSION 3.5)

project (tkDNN)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tkDNN)

# project specific flags
if(DEBUG)
    add_definitions(-DDEBUG)
endif()


#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDA 9.0 REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 --compiler-options '-fPIC'")

find_package(CUDNN REQUIRED)

# compile
file(GLOB tkdnn_CUSRC "src/kernels/*.cu")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})
cuda_add_library(kernels SHARED ${tkdnn_CUSRC})


#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

include_directories(/usr/include/gdal)

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS})

file(GLOB class_SRC "src/class_src/*.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -O3")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES} "~/repos/cereal/include" ${CMAKE_CURRENT_SOURCE_DIR}/tracker_CLASS/c++/src /usr/include/python2.7)

set(class_LIBS ${OpenCV_LIBS} -lgdal yaml-cpp python2.7)

add_library(tkDNN SHARED ${tkdnn_SRC})
target_link_libraries(tkDNN ${tkdnn_LIBS})

add_library(CLASS SHARED ${class_SRC})
target_link_libraries(CLASS ${class_LIBS})


#static 
#add_library(tkDNN_static STATIC ${tkdnn_SRC})
#target_link_libraries(tkDNN_static ${tkdnn_LIBS})

add_executable(test_simple tests/simple/test_simple.cpp)
target_link_libraries(test_simple tkDNN)

add_executable(test_mnist tests/mnist/test_mnist.cpp)
target_link_libraries(test_mnist tkDNN)

add_executable(test_mnistRT tests/mnist/test_mnistRT.cpp)
target_link_libraries(test_mnistRT tkDNN)

## YOLO NETS
add_executable(test_yolo tests/yolo/yolo.cpp)
target_link_libraries(test_yolo tkDNN)

add_executable(test_yolo_voc tests/yolo_voc/yolo_voc.cpp)
target_link_libraries(test_yolo_voc tkDNN)

add_executable(test_yolo_tiny tests/yolo_tiny/yolo_tiny.cpp)
target_link_libraries(test_yolo_tiny tkDNN)

add_executable(test_yolo_relu tests/yolo_relu/yolo_relu.cpp)
target_link_libraries(test_yolo_relu tkDNN)


add_executable(test_yolo_224 tests/yolo_224/yolo_224.cpp)
target_link_libraries(test_yolo_224 tkDNN)

add_executable(test_yolo_berkeley tests/yolo_berkeley/yolo_berkeley.cpp)
target_link_libraries(test_yolo_berkeley tkDNN)

add_executable(test_yolo3_coco4 tests/yolo3_coco4/yolo3_coco4.cpp)
target_link_libraries(test_yolo3_coco4 tkDNN)

add_executable(test_yolo3_berkeley tests/yolo3_berkeley/yolo3_berkeley.cpp)
target_link_libraries(test_yolo3_berkeley tkDNN)

add_executable(test_yolo3_tetrapack tests/yolo3_tetrapack/yolo3_tetrapack.cpp)
target_link_libraries(test_yolo3_tetrapack tkDNN)

add_executable(test_yolo3_tetrapack_resize tests/yolo3_tetrapack_resize/yolo3_tetrapack_resize.cpp)
target_link_libraries(test_yolo3_tetrapack_resize tkDNN)

add_executable(test_yolo3_BCDS6 tests/yolo3_BCDS6/yolo3_BCDS6.cpp)
target_link_libraries(test_yolo3_BCDS6 tkDNN)
add_executable(test_yolo3_flir tests/yolo3_flir/yolo3_flir.cpp)
target_link_libraries(test_yolo3_flir tkDNN)

add_executable(test_imuodom tests/imuodom/imuodom.cpp)
target_link_libraries(test_imuodom tkDNN)
################################################################################


add_executable(test_rtinference tests/test_rtinference/rtinference.cpp)
target_link_libraries(test_rtinference tkDNN)

add_executable(yolo3_demo demo/demo/demo.cpp 
                        tracker_CLASS/c++/src/ekf.cpp
                        tracker_CLASS/c++/src/trackutils.cpp
                        tracker_CLASS/c++/src/plot.cpp
                        tracker_CLASS/c++/src/tracker.cpp )

                        
target_link_libraries(yolo3_demo tkDNN CLASS)




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


#-------------------------------------------------------------------------------
# Prepare for test
#-------------------------------------------------------------------------------
set(TEST_DATA true CACHE BOOL "If true download deps")
if( ${TEST_DATA} )
	message("Launching pre-build dependency installer script...")

    execute_process (COMMAND bash -c "bash build_models.sh download"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

	set(TEST_DATA false CACHE BOOL "If true download deps" FORCE)
	message("Finished dowloading test weights")
endif()

