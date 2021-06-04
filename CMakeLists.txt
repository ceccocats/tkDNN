cmake_minimum_required(VERSION 3.5)
set(PROJ_NAME tkDNN)
project (tkDNN)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
if(UNIX)

####
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wno-deprecated-declarations -Wno-unused-variable ")
endif()
if(WIN32)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "/O2 /FS /EHsc")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
#add extras for baggage
endif(WIN32)
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
target_link_libraries(kernels ${CUDA_CUBLAS_LIBRARIES})


#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

# gives problems in cross-compiling, probably malformed cmake config
find_package(yaml-cpp REQUIRED)

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES})
add_library(tkDNN SHARED ${tkdnn_SRC})
target_link_libraries(tkDNN ${tkdnn_LIBS})
####compile
#set(PROJ_NAME BaggageAIApi)
# Path to BaggageAI project folder.
set(BAGGAGEAI_PATH /home/baggageai/files)
# Give a custom name to shared library which is provided by DIMENSIONLESS.
#set(BAGGAGEAI_LIB_NAME libBaggageAI)
# Define C++ level, could be 11 or 17 as well.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# Define compiler optimization level.
set(CMAKE_CXX_FLAGS "-O3")
# Do print warnings uppon compilation, let's keep our code as clean as possible.
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# Apply flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DBOOST_LOG_DYN_LINK")
set(Casablanca_LIBRARIES "-lboost_log -lboost_log_setup -lboost_thread -lboost_system -lcrypto -lssl -lcpprest -lpthread")


# Note: We do not recommend using GLOB or GLOB_RECURSE to collect a list of source files from your source tree.
# If no CMakeLists.txt file changes when a source is added or removed then the generated build system cannot know
# when to ask CMake to regenerate.

file(GLOB_RECURSE SOURCE_FILES "main.cpp" "handler.cpp" "src/*.cpp")

add_executable(baggageAPI ${SOURCE_FILES})
set(Casablanca_LIBRARIES "-lboost_log -lboost_log_setup -lboost_thread -lboost_system -lcrypto -lssl -lcpprest -lpthread" )
set(tkdnn_LIBS kernels ${Casablanca_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# Link BaggageAI library' include folder.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES} ${Casablanca_LIBRARIES} ${CMAKE_CXX_FLAGS})
# Define BaggageAI library' shared library.
#add_library(${BAGGAGEAI_LIB_NAME} SHARED IMPORTED)
# Set a path to BaggageAI library' shared library
#set_property(TARGET ${BAGGAGEAI_LIB_NAME} PROPERTY IMPORTED_LOCATION "${BAGGAGEAI_PATH}/libBaggageAI.so")

# Link all libraries together.
target_link_libraries(baggageAPI ${tkdnn_LIBS})

#static 
#add_executable(demo demo/inf.cpp)
#target_link_libraries(demo tkDNN)

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

