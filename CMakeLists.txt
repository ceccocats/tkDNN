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

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

include_directories(/usr/include/gdal)

# compile Discovery only if TensorRT is installed
find_library(NVINFER NAMES nvinfer)
if(NVINFER STREQUAL "NVINFER-NOTFOUND")
    set(NVINFER_INCLUDES "/usr/local/nvidia/tensorrt/include/")
    link_directories(/usr/local/nvidia/tensorrt/targets/x86_64-linux-gnu/lib/ 
		     /usr/local/cuda/targets/x86_64-linux/lib/)
endif()

file(GLOB tkdnn_CUSRC "src/kernels/*.cu")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${NVINFER_INCLUDES})
cuda_add_library(kernels SHARED ${tkdnn_CUSRC})

file(GLOB tkdnn_SRC "src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} -lcudnn -lnvinfer ${OpenCV_LIBS} -lgdal)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES} "~/repos/cereal/include" ${CMAKE_CURRENT_SOURCE_DIR}/tracker_CLASS/c++/src)
include_directories( BEFORE ${MY_SOURCE_DIR}/src /usr/include/python2.7 )
add_library(tkDNN SHARED ${tkdnn_SRC})
target_link_libraries(tkDNN ${tkdnn_LIBS})

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
################################################################################


add_executable(test_rtinference tests/test_rtinference/rtinference.cpp)
target_link_libraries(test_rtinference tkDNN)

add_executable(yolo3_demo demo/demo/demo.cpp 
                        tracker_CLASS/c++/src/ekf.cpp
                        tracker_CLASS/c++/src/trackutils.cpp
                        tracker_CLASS/c++/src/plot.cpp
                        tracker_CLASS/c++/src/tracker.cpp )

                        
target_link_libraries(yolo3_demo tkDNN)
target_link_libraries(yolo3_demo python2.7)

add_executable(class_server demo/server/server_less_dummy.cpp)
target_link_libraries(class_server pthread tkDNN )


#install
#if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" 
#         CACHE PATH "default install path" FORCE)
#endif()
message("install dir:" ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY include/ DESTINATION include/${CMAKE_PROJECT_NAME}
FILES_MATCHING PATTERN "*.h")
install(TARGETS tkDNN kernels DESTINATION lib)
install(FILES "${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME}Config.cmake" # source directory
        DESTINATION "share/${CMAKE_PROJECT_NAME}/cmake/" # target directory
)
