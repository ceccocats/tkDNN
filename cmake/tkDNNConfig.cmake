message("-- Found tkDNN")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -fPIC")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_library(NVINFER NAMES nvinfer)
if(NVINFER STREQUAL "NVINFER-NOTFOUND")
    set(NVINFER_INCLUDES "/usr/local/nvidia/tensorrt/include/")
    link_directories(/usr/local/nvidia/tensorrt/targets/x86_64-linux-gnu/lib/ 
		     /usr/local/cuda/targets/x86_64-linux/lib/)
endif()

set(tkDNN_INCLUDE_DIRS 
	${CUDA_INCLUDE_DIRS} 
	${OPENCV_INCLUDE_DIRS} 
    ${NVINFER_INCLUDES}
)

set(tkDNN_LIBRARIES 
    tkDNN 
    kernels 
    ${CUDA_LIBRARIES} 
    ${CUDA_CUBLAS_LIBRARIES} 
    -lcudnn 
    -lnvinfer 
    ${OpenCV_LIBS}
)

set(tkDNN_FOUND true)
