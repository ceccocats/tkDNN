# find the library
if(CUDA_FOUND)
  find_cuda_helper_libs(cudnn)
  set(CUDNN_LIBRARY ${CUDA_cudnn_LIBRARY} CACHE FILEPATH "location of the cuDNN library")
  unset(CUDA_cudnn_LIBRARY CACHE)

  find_cuda_helper_libs(nvinfer)
  set(NVINFER_LIBRARY ${CUDA_nvinfer_LIBRARY} CACHE FILEPATH "location of the nvinfer library")
  unset(CUDA_nvinfer_LIBRARY CACHE)
endif()

# find the include
if(CUDNN_LIBRARY)
  find_path(CUDNN_INCLUDE_DIR
    cudnn.h
    PATHS ${CUDA_TOOLKIT_INCLUDE}
    DOC "location of cudnn.h"
    NO_DEFAULT_PATH
  )

  if(NOT CUDNN_INCLUDE_DIR)
    find_path(CUDNN_INCLUDE_DIR
      cudnn.h
      DOC "location of cudnn.h"
    )
  endif()

  message("-- Found CUDNN: " ${CUDNN_LIBRARY})
  message("-- Found CUDNN include: " ${CUDNN_INCLUDE_DIR})
endif()

if(NVINFER_LIBRARY)
  find_path(NVINFER_INCLUDE_DIR
    NvInfer.h
    PATHS ${CUDA_TOOLKIT_INCLUDE}
    DOC "location of NvInfer.h"
    NO_DEFAULT_PATH
  )

  if(NOT NVINFER_INCLUDE_DIR)
    find_path(NVINFER_INCLUDE_DIR
        NvInfer.h
        DOC "location of NvInfer.h"
    )
  endif()

  message("-- Found NVINFER: " ${NVINFER_LIBRARY})
  message("-- Found NVINFER include: " ${NVINFER_INCLUDE_DIR})
endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
  FOUND_VAR CUDNN_FOUND
  REQUIRED_VARS
  CUDNN_LIBRARY
  CUDNN_INCLUDE_DIR
  VERSION_VAR CUDNN_VERSION
)

if(CUDNN_FOUND)
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY} ${NVINFER_LIBRARY})
  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR} ${NVINFER_INCLUDE_DIR})
endif()

set(CUDNN_FOUND true)