# Find the header files

find_path(CUDNN_INCLUDE_DIR
        ${CMAKE_SYSROOT}/usr/local/include
        ${CMAKE_SYSROOT}/usr/include
        /usr/local/nvidia/tensorrt/include/
        NO_DEFAULT_PATH
)

set(OLD_ROOT ${CMAKE_FIND_ROOT_PATH})
list(APPEND CMAKE_FIND_ROOT_PATH /)
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.7)
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.5)
find_library(CUDNN_LIB
    NAMES cudnn
    PATHS
    /usr/local/driveworks/targets/${CMAKE_SYSTEM_PROCESSOR}-Linux/lib
    /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/
    NO_DEFAULT_PATH
)
find_library(CUDNN_NVLIB
    NAMES "nvinfer"
    PATHS
    /usr/local/driveworks/targets/${CMAKE_SYSTEM_PROCESSOR}-Linux/lib
    /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/
    NO_DEFAULT_PATH
)
set(CMAKE_FIND_ROOT_PATH ${OLD_ROOT})

set(CUDNN_LIBRARIES ${CUDNN_LIB} ${CUDNN_NVLIB})
message("-- Found CUDNN: "  ${CUDNN_LIB})
message("-- Found NVINFER: "  ${CUDNN_NVLIB})
set(CUDNN_FOUND true)
