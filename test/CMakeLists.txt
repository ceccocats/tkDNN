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
#-------------------------------------------------------------------
#-------------------------------------------------------------------------------
# External Libraries
#------------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")

# gives pr
#-------------------------------------------------------------------------------
# Build Libraries
#----------------------------------------------------------------------------
file(GLOB_RECURSE SOURCE_FILES "client_img.cpp")

add_executable(client ${SOURCE_FILES})
#set(Casablanca_LIBRARIES "-lboost_log -lboost_log_setup -lboost_thread -lboost_system -lcrypto -lssl -lcpprest -lpthread" )
set(tkdnn_LIBS ${OpenCV_LIBS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# Link BaggageAI library' include folder.
include_directories(${OPENCV_INCLUDE_DIRS} ${CMAKE_CXX_FLAGS})
# Define BaggageAI library' shared library.
#add_library(${BAGGAGEAI_LIB_NAME} SHARED IMPORTED)
# Set a path to BaggageAI library' shared library
#set_property(TARGET ${BAGGAGEAI_LIB_NAME} PROPERTY IMPORTED_LOCATION "${BAGGAGEAI_PATH}/libBaggageAI.so")

# Link all libraries together.
target_link_libraries(client ${tkdnn_LIBS})

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
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" # source directory
        DESTINATION "share/tkDNN/cmake/" # target directory
)

